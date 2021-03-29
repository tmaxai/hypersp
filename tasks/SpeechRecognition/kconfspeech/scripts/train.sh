#!/bin/bash

DATA_DIR=${1:-${DATA_DIR:-"data/KconfSpeech"}}
MODEL_CONFIG=${2:-${MODEL_CONFIG:-"/configs/jasper10x5dr_sp_offline_specaugment.yaml"}}
RESULT_DIR=${3:-${RESULT_DIR:-"results"}}
CHECKPOINT=${4:-${CHECKPOINT:-""}}
CREATE_LOGFILE=${5:-${CREATE_LOGFILE:-"true"}}
CUDNN_BENCHMARK=${6:-${CUDNN_BENCHMARK:-"true"}}
NUM_GPUS=${7:-${NUM_GPUS:-1}}
AMP=${8:-${AMP:-"true"}}
EPOCHS=${9:-${EPOCHS:-200}}
SEED=${10:-${SEED:-6}}
BATCH_SIZE=${11:-${BATCH_SIZE:-256}}
LEARNING_RATE=${12:-${LEARNING_RATE:-"0.01"}}
GRADIENT_ACCUMULATION_STEPS=${13:-${GRADIENT_ACCUMULATION_STEPS:-8}}
EMA=${EMA:-0.0}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-5}
TASK_PATH=${TASK_PATH:-"tasks.SpeechRecognition.kconfspeech.local.manifest"}
VOCAB=${VOCAB:-"vocab"}

mkdir -p "$RESULT_DIR"

export OMP_NUM_THREADS=8

CMD="python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=54321"
CMD+=" ../../../train.py"
CMD+=" --task_path=$TASK_PATH"
CMD+=" --batch_size=$BATCH_SIZE"
CMD+=" --num_epochs=$EPOCHS"
CMD+=" --output_dir=$RESULT_DIR"
CMD+=" --model_cfg=$PWD$MODEL_CONFIG"
CMD+=" --lr=$LEARNING_RATE"
CMD+=" --ema=$EMA"
CMD+=" --seed=$SEED"
CMD+=" --optimizer=novograd"
CMD+=" --dataset_dir=$DATA_DIR"
CMD+=" --val_manifest=$DATA_DIR/valid-wav.json"
CMD+=" --train_manifest=$DATA_DIR/D20-wav.json"
CMD+=",$DATA_DIR/D21-wav.json"
CMD+=",$DATA_DIR/D22-wav.json"
CMD+=",$DATA_DIR/D23-wav.json"
CMD+=",$DATA_DIR/D24-wav.json"
CMD+=",$DATA_DIR/D25-wav.json"
CMD+=",$DATA_DIR/D26-wav.json"
CMD+=",$DATA_DIR/D27-wav.json"
CMD+=" --weight_decay=1e-3"
CMD+=" --save_freq=$SAVE_FREQUENCY"
CMD+=" --eval_freq=100"
CMD+=" --train_freq=1"
CMD+=" --lr_decay"
CMD+=" --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
CMD+=" --num_gpus=$NUM_GPUS"
CMD+=" --vocab=$VOCAB"

[ "$AMP" == "true" ] && \
CMD+=" --amp"
[ "$CUDNN_BENCHMARK" = "true" ] && \
CMD+=" --cudnn"
[ -n "$CHECKPOINT" ] && \
CMD+=" --ckpt=${CHECKPOINT}"

if [ "$CREATE_LOGFILE" = "true" ] ; then
   export GBS=$(expr $BATCH_SIZE \* $NUM_GPUS)
   printf -v TAG "jasper_train_benchmark_amp-%s_gbs%d" "$AMP" $GBS
   DATESTAMP=`date +'%y%m%d%H%M%S'`
   LOGFILE=$RESULT_DIR/$TAG.$DATESTAMP.log
   printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi
set +x
