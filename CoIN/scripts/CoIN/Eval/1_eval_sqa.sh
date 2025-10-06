#!/bin/bash

python ./instruct/LoRASelect.py --codebook ./instruct/codebooks/ --instruct ./playground/Instructions/ScienceQA/test.json

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='CIA_Task8_Type1'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/Instruction/Only_Pretrain_1.5_CIA/ScienceQA/llava-1.5-7b-lora'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/CoIN/ScienceQA"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CoIN.model_vqa_science \
        --model-path $MODELPATH \
        --model-base ./checkpoints/Vicuna/vicuna-7b-v1.5 \
        --question-file ./playground/Instructions/ScienceQA/test.json \
        --image-folder ./cl_dataset \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/CoIN/eval_science_qa.py \
    --base-dir ./cl_dataset/ScienceQA \
    --result-file $output_file \
    --output-file $RESULT_DIR/$STAGE/output.jsonl \
    --output-result $RESULT_DIR/$STAGE/output_result.jsonl \

#python llava/eval/CoIN/create_prompt.py \
#    --rule llava/eval/CoIN/rule.json \
#    --questions ./playground/Instructions/ScienceQA/test.json \
#    --results $output_file \