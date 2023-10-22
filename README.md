# NLP 실습 모음

## Preprocessing

After preprocess your data, you may need to (symbolic) link on each task directory.

```
./preprocessing/regex.ipynb
./preprocessing/convert.ipynb
```

## Tokenization

### Tokenization with Mecab

```
./tokenization/mecab.ipynb
```

### Tokenization with Huggingface Tokenizer

```
./tokenization/hf_tokenizer.ipynb
```

## RNN Text Classification

### Train

```sh
$ python train.py \
    --model_name nsmc-lstm \
    --train_tsv_fn ./data/ratings_train.train.tsv \
    --valid_tsv ./data/ratings_train.valid.tsv \
    --test_tsv_fn ./data/ratings_test.tsv \
    --gpu_id 0 \
    --n_epochs 40 \
    --lr 1e-4
```

### Classify

```sh
cat ./data/ratings_test.tsv | cut -f2 | shuf | head -n 20 | python ./classify.py \
    --model_fn ./checkpoints/nsmc-lstm-20231005-165735.pt \
    --device 0
```

## BERT Text Classification

### Train

```sh
$ python finetune.py \
    --model_name nsmc-bert \
    --train_tsv_fn ./data/ratings_train.train.tsv \
    --valid_tsv_fn ./data/ratings_train.valid.tsv \
    --test_tsv_fn ./data/ratings_test.tsv \
    --batch_size_per_device 32 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --fp16
```

### Classify

```sh
$ cat ./data/ratings_test.tsv | cut -f2 | shuf | head -n 20 | /usr/bin/python3.9 ./classify.py \
    --checkpoint_dir /home/ubuntu/nlp-practice/bert_text_classification/checkpoints/nsmc-bert-20231017-114830/checkpoint-1008 \
    --device 0
```

## Neural Machine Translation with Huggingface

### Train Tokenizer

```sh
$ python3 train_tokenizer.py \
    --train_files ./data/nmt_corpus/train.en ./data/nmt_corpus/train.ko \
    --output_name nmt_corpus \
    --vocab_size 30000
```

### Train

```sh
$ python train.py \
    --model_name nmt-enko \
    --tokenizer_dir_path ./tokenizer/sample \
    --data_dir_path ./data/sample \
    --src_lang en \
    --tgt_lang ko \
    --num_train_epochs 50 \
    --batch_size_per_device 64 \
    --gradient_accumulation_steps 8 \
    --fp16
```

### Translate

```sh
$ cat ./data/sample/valid.en | shuf | head -n 10 | python ./translate.py \
    --model_name test \
    --tokenizer_name sample \
    --gpu_id 0
```