MODEL="model name for watermarked prompts"
TEST_MODEL="model to eval the watermarked prompts, usually same as MODEL"

MODEL_PATH="model path"
TEST_MODEL_PATH="test model's path"

PROMPT_PATH="path to original prompts"  # demo is prompts/
W_PROMPT_PATH="path to save watermarked prompts"
EVAL_RESULT_PATH="path to save eval results"

FILTER_PATH="list[token_id] saved in pickle for filter tokens"

# You can use following code to match different task (dataset)
TASK="qa"

if [ $TASK == "qa" ]; then
  TRAIN_DATA="datasets/standard_bigbench.json"
elif [ $TASK == "math" ]; then
  TRAIN_DATA="datasets/standard_gsm8k.json"
fi

SAVE_PATH_WP="${W_PROMPT_PATH}/${TASK}_${MODEL}.json"
SAVE_PATH_RESULT="${EVAL_RESULT_PATH}/${TASK}_${MODEL}_${TEST_MODEL}.json"

python -u embedding.py \
    --model_path $MODEL_PATH \
    --filter_path $FILTER_PATH \
    --original_prompts "$PROMPT_PATH/$TASK.json" \
    --prompt_train_data $TRAIN_DATA \
    --num_fb 4 \
    --num_at 8 \
    --num_cv 5 \
    --num_vf 5 \
    --num_sm 3 \
    --ss 2 \
    --top_k 100 \
    --max_epoch 10 \
    --p_f 0.5 \
    --p_d 0.5 \
    --save_file $SAVE_PATH_WP

python -u overall_eval.py \
    --model_path $TEST_MODEL_PATH \
    --prompts_file "$W_PROMPT_PATH/${TASK}_${MODEL}.json" \
    --prompt_train_data $TRAIN_DATA \
    --eval_data_num 100 \
    --output_file $SAVE_PATH_RESULT \
    --temperature 1.0 \
    --top_p 0.9 \
    --max_new_tokens 256 \
    --batch_size 32 \
    --if_effectiveness True \