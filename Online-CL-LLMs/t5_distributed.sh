
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

deepspeed --num_gpus=2 src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir configs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/task1572_samsum_summary \
   --output_dir logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
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
   --trans_hidden_dim 100 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary/checkpoint*


deepspeed --num_gpus=2 src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --previous_lora_path logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/task363_sst2_polarity_classification \
   --output_dir logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-task363_sst2_polarity_classification \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
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
   --metric_for_best_model eval_rougeL_for_task363_sst2_polarity_classification \
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
   --trans_hidden_dim 100 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-task363_sst2_polarity_classification/checkpoint*



deepspeed --num_gpus=2 src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --previous_lora_path logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-task363_sst2_polarity_classification/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-task363_sst2_polarity_classification/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/task1290_xsum_summarization \
   --output_dir logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-task1290_xsum_summarization \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
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
   --metric_for_best_model eval_rougeL_for_task1290_xsum_summarization \
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
   --trans_hidden_dim 100 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-task1290_xsum_summarization/checkpoint*



deepspeed --num_gpus=2 src/run_t5_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --previous_lora_path logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-task1290_xsum_summarization/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-task1290_xsum_summarization/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/task181_outcome_extraction \
   --output_dir logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-task181_outcome_extraction \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
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
   --metric_for_best_model eval_rougeL_for_task181_outcome_extraction \
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
   --trans_hidden_dim 100 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-task181_outcome_extraction/checkpoint*






deepspeed --num_gpus=2 src/run_t5_new_eval.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5p-220m \
   --previous_lora_path logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-task1729_personachat_generate_next/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-task1590_diplomacy_text_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-task591_sciq_answer_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/14-task1687_sentiment140_classification/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-task1572_samsum_summary/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-task363_sst2_polarity_classification/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-task1290_xsum_summarization/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-task181_outcome_extraction/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-task002_quoref_answer_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-task1510_evalution_relation_extraction/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-task639_multi_woz_user_utterance_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-task1729_personachat_generate_next/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-task073_commonsenseqa_answer_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-task1590_diplomacy_text_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-task748_glucose_reverse_cause_event_detection/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-task511_reddit_tifu_long_text_summarization/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-task591_sciq_answer_generation/saved_weights,logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/14-task1687_sentiment140_classification/saved_weights \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/task875_emotion_classification \
   --output_dir logs_and_outputs/test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/15-task875_emotion_classification \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-04 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name test_llama_7b_superni_our_8_1_4_L2_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
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
   --metric_for_best_model eval_rougeL_for_task875_emotion_classification \
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
   --trans_hidden_dim 100 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N
