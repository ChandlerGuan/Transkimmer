# choose from 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli', 'imdb'
TASK_NAME=imdb

SEED=42
SKIM_FACTOR=0.5

if [ $TASK_NAME == wnli ] || [ $TASK_NAME == mrpc ]
then
    EPOCH_NUM=5
else
    EPOCH_NUM=3
fi

OUTPUT_DIR=model/glue/${TASK_NAME}/

if [ -d "$OUTPUT_DIR" ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
fi

mkdir -p ${OUTPUT_DIR}

python -u src/run_glue_no_trainer.py \
  --model_type transkimer \
  --skim_coefficient ${SKIM_FACTOR} \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --do_train \
  --do_evaluate \
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --seed ${SEED} \
  --num_train_epochs ${EPOCH_NUM} \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log.log