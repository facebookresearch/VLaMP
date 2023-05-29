#!/bin/bash

set -x
export DRY_RUN=0
export DATA_DIR=.data # path relative to project root
export ROOT_DIR=. # try to give path the path relative to project root
export CUDA_DEVICE="0" # -1 for cpu
export TSL_DEBUG="false"
export join_train_validation="true"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:100"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


MODELDIR=".models/yyaqgtoq"
MODEL="$MODELDIR/model.tar.gz"
DATA=".data/crosstask/test.txt"
OUTPUT="$MODELDIR/eval.json"


DEVICE_MAP=\
"{\"0\": [0], \
\"1\": [1, 2, 3], \
\"2\": [4, 5, 6], \
\"3\": [7,8], \
\"4\": [9,10,11]}"

allennlp evaluate $MODEL $DATA \
--output-file=$OUTPUT \
--include-package=vlamp \
--overrides "{\"model.device_map\":$DEVICE_MAP,\"model.per_node_beam_size\":3, \"model.predict_mode\":false, \"model.top_k\": 10, \"model.max_steps\":4,\"model.min_steps\": 4, \"validation_dataset_reader.minimum_target_length\":4,\"validation_dataset_reader.file_reader.segments_file\":\"videoclip_preds.json\" }" \
--cuda-device "$CUDA_DEVICE"

set +x