#!/bin/bash

export GCP_PROJECT_NAME=bigbird-freefly
export GCP_EXP_BUCKET=gs://bigbird-freefly/
# TF_XLA_FLAGS=--tf_xla_auto_jit=2
python3 bigbird/summarization/run_summarizationb.py \
  --data_dir="$GCP_EXP_BUCKET"summarization/FNS/4.0.0 \
  --output_dir="$GCP_EXP_BUCKET"summarization/FNS_plargeb_512_8.0 \
  --attention_type=block_sparse \
  --couple_encoder_decoder=False \
  --max_encoder_length=3072 \
  --max_decoder_length=512 \
  --num_attention_heads=16 \
  --num_hidden_layers=16 \
  --hidden_size=1024 \
  --intermediate_size=4096 \
  --block_size=64 \
  --scope=pegasus \
  --norm_type=prenorm \
  --hidden_act=relu \
  --use_bias=False \
  --rescale_embedding=True \
  --vocab_model_file=pegasus \
  --substitute_newline="" \
  --train_batch_size=1 \
  --eval_batch_size=1 \
  --do_train=True \
  --do_eval=True \
  --use_tpu=True \
  --tpu_name=tpu_eu_3b \
  --tpu_zone=europe-west4-a \
  --gcp_project="$GCP_PROJECT_NAME" \
  --num_tpu_cores=8\
  --init_checkpoint=gs://bigbird-transformer/summarization/pubmed/pegasus/model.ckpt-0