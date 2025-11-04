
#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
#SBATCH -p compute 
#SBATCH -N 1                           
#SBATCH -t 20:00:00   
#SBATCH --mem 128G 
#SBATCH --gres=gpu:a100-sxm4-80gb:1

fuser -k /dev/nvidia*

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
port=$(shuf -i25000-30000 -n1)  

python3 src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP \
   --task_config_dir configs/CodeTask/CONCODE \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 256 \
   --generation_max_length 256 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 1 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/checkpoint*


python3 src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/CodeTrans \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 256 \
   --generation_max_length 256 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_CodeTrans \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 1 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/checkpoint*



python3 src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/CodeSearchNet \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 256 \
   --generation_max_length 256 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_CodeSearchNet \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 1 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/checkpoint*



python3 src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/BFP \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 256 \
   --generation_max_length 256 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_BFP \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 1 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/checkpoint*






python3 src/run_t5_new_eval.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --previous_lora_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-CONCODE/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-CodeTrans/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-CodeSearchNet/saved_weights,logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-BFP/saved_weights \
   --data_dir CODETASK_Benchmark \
   --task_order CONCODE,CodeTrans,CodeSearchNet,BFP \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/CodeTask/task875_emotion_classification \
   --output_dir logs_and_outputs/test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/15-task875_emotion_classification \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 1 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --run_name test_t5_codetask_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way L2 \
   --max_source_length 1024 \
   --max_target_length 256 \
   --generation_max_length 256 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_BFP \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 1 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N
