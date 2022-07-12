SEED=42
SKIM_FACTOR=0.3
EPOCH_NUM=5
BATCH_SIZE=64

OUTPUT_DIR=model/squad

if [ -d "$OUTPUT_DIR" ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
fi

mkdir -p ${OUTPUT_DIR}

python -u src/run_qa_no_trainer.py \
  --model_type transkimer \
  --skim_coefficient ${SKIM_FACTOR} \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --dataset_name squad_v2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative true \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --do_train \
  --do_evaluate \
  --seed ${SEED} \
  --num_train_epochs ${EPOCH_NUM} \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log.log