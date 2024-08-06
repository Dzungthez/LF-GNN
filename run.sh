#!/bin/bash
# source activate myenv
# Default parameters
INPUT_PATH="data/longformer_data/input_longformer"
LABELS_PATH="data/longformer_data/full_data_labels_test.json"
FILE_WEIGHT_MATRIX="data/gcn_input/lawyers/weight_matrix_lids_1.csv"
FILE_MATRIX_PATH="data/gcn_input/lawyers/adjacency_matrix_lids_1.csv"
CASE_META_PATH="data/gcn_input/case_meta"
BATCH_SIZE=2
LR=1e-5
NUM_EPOCHS=12
CHECKPOINT_DIR="checkpoints"
LOAD_FROM_CHECKPOINT=true
TRAINING=true

# Run the Python script with the default parameters
python3 main.py \
  --input_path $INPUT_PATH \
  --labels_path $LABELS_PATH \
  --file_weight_matrix $FILE_WEIGHT_MATRIX \
  --file_matrix_path $FILE_MATRIX_PATH \
  --case_meta_path $CASE_META_PATH \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --num_epochs $NUM_EPOCHS \
  --checkpoint_dir $CHECKPOINT_DIR \
  $(if $LOAD_FROM_CHECKPOINT; then echo "--load_from_checkpoint"; fi) \
  $(if $TRAINING; then echo "--training"; fi)
