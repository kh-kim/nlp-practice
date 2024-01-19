SRC_LANG=en
TGT_LANG=ko
MODEL_NAME=nmt_corpus-${SRC_LANG}${TGT_LANG}

TOKENIZER_DIR_PATH=./tokenizers/sample
DATA_DIR_PATH=./data/sample

python3.9 train.py \
    --model_name ${MODEL_NAME} \
    --tokenizer_dir_path ${TOKENIZER_DIR_PATH} \
    --data_dir_path ${DATA_DIR_PATH} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --num_train_epochs 50 \
    --batch_size_per_device 64 \
    --gradient_accumulation_steps 4 \
    --fp16
