# nlp-practice

```sh
python train.py --model_name nsmc-lstm --train_tsv_fn ./data/ratings_train.train.tsv --valid_tsv ./data/ratings_train.valid.tsv --test_tsv_fn ./data/ratings_test.tsv --gpu_id 0 --n_epochs 40 --lr 1e-4
```

```sh
$ python finetune.py --model_name nsmc-bert --train_tsv_fn ./data/ratings_train.train.tsv --valid_tsv_fn ./data/ratings_train.valid.tsv --test_tsv_fn ./data/ratings_test.tsv --batch_size_per_device 32 --gradient_accumulation_steps 8 --num_train_epochs 3 --fp16
```

