# choose from 'yelp' '20news'
TASK_NAME=20news

SEED=42
SKIM_FACTOR=0.2

if [ $TASK_NAME == 20news ] 
then
    EPOCH_NUM=5
else
    EPOCH_NUM=2
fi

OUTPUT_DIR=model/classification/${TASK_NAME}/

if [ -d "$OUTPUT_DIR" ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
fi

mkdir -p ${OUTPUT_DIR}

python3 src/run_classification.py  \
  --task_name ${TASK_NAME}  \
  --model_type transkimer  \
  --skim_coefficient ${SKIM_FACTOR} \
  --model_name_or_path bert-base-uncased \
  --data_dir dataset/${TASK_NAME} \
  --max_seq_length 512  \
  --per_gpu_train_batch_size 32  \
  --per_gpu_eval_batch_size 16 \
  --save_steps 300 \
  --learning_rate 3e-5 \
  --save_steps 2000  \
  --num_train_epochs ${EPOCH_NUM}  \
  --output_dir ${OUTPUT_DIR} \
  --do_lower_case  \
  --do_eval  \
  --do_train \
  --evaluate_during_training \
  --overwrite_output_dir    2>&1 | tee ${OUTPUT_DIR}/log.log