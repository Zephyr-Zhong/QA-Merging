set -ex

export CUDA_VISIBLE_DEVICES="0"
# =============== Evaluation Prompt Template ===============
# qwen25 reasoning: qwen25-math-cot; Deepseek reasoning: deepseek-math-cot; Base model: cot;
# qwen3 thinking reasoning: "qwen3-think"; qwen3-instruct: "qwen3-instruct-2507"

PROMPT_TYPE="qwen3-think"
# PROMPT_TYPE="qwen3-instruct-2507"
# ==========================================================


# =============== Model Path and Output Path ===============
# == base: thinking/instruct
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Thinking-2507"
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Instruct-2507"
# == RPAM
MODEL_NAME_OR_PATH="MERGED_MODEL_PATH"


# == base: Thinking/Instruct
# MODEL_NAME_FOR_SAVE="Qwen3-4B-Thinking-2507"
# MODEL_NAME_FOR_SAVE="Qwen3-4B-Instruct-2507"
# == RPAM
MODEL_NAME_FOR_SAVE=qwen3_RPAM/contrastive_w_1000_MI_0.5/50-0.001-64-16
# ==========================================================


# ===============       Datasets        ===============
# DATASETS:aime24,math,gsm8k,college_math,minerva_math,olympiadbench,gpqa_diamond
# ==========================================================


# ===============       Eval parameters         ===============
SEED=41
NUM_SHOTS=0     # CoT few shots setting
OUTPUT_DIR="./math_eval/${MODEL_NAME_FOR_SAVE}/"
SPLIT="test"
NUM_TEST_SAMPLE=-1
MAX_NEW_TOKEN=32768
# ==========================================================


# ==================== Evaluation ====================
# "gsm8k,math,minerva_math,gpqa_diamond,aime24,aime25"
MAX_NEW_TOKEN=32768
DATASETS="gsm8k,math,aime24"
N_SAMPLING=1

DATA_NAME=${DATASETS}
TOKENIZERS_PARALLELISM=false \
python3 -u ./math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature 0.6 \
    --n_sampling ${N_SAMPLING} \
    --top_p 0.95 \
    --top_k 20 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --num_shots ${NUM_SHOTS} \
    --save_outputs \
    --overwrite \
    --max_tokens_generate ${MAX_NEW_TOKEN}
# # # ==========================================================

# ==================== Response Statistics =================
DATA_PROCESS_OUTPUT_DIR="./outputs/${OUTPUT_DIR}/math_eval_${MAX_NEW_TOKEN}"
python3 ./data_process.py ${MODEL_NAME_OR_PATH} ${DATA_PROCESS_OUTPUT_DIR}
# ==========================================================                                                    