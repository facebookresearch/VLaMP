#!/bin/bash

set -x
export DRY_RUN=0
export DATA_DIR=.data # path relative to project root
export ROOT_DIR=. # try to give path the path relative to project root
export CUDA_DEVICE=1 # -1 for cpu
export TSL_DEBUG="false"
export join_train_validation="true"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:100"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export s3d_features="true"
export i3d_features="false"
export audio_features="false"
export resnet_features="false"
export dataset_name="\"crosstask_closed_domain_gpt2\""
export random_lm_weights="false"
export freeze_lm="false"
export attention_dropout="2"
export hidden_dropout="2"
export lm="\"gpt2\""
export observation_type="\"pre\""
export zero_shot="false"


CONFIG_FILE="model_configs/closed_domain_gpt2.jsonnet"
MODELDIR=".models/2q0ab584"
MODEL="$MODELDIR/model.tar.gz"
DATA=".data/crosstask/test.txt"
OUTPUT="$MODELDIR/predictions.json"

allennlp predict $MODEL $DATA \
--output-file=$OUTPUT \
--include-package=vlamp \
--overrides "{\"model.predict_mode\":true, \"model.top_k\": 10}" \
--batch-size=1 \
--use-dataset-reader \
--cuda-device $CUDA_DEVICE

set +x