#!/bin/bash

DATA_DIR=${1:-${DATA_DIR:-"/data/KtelSpeech"}}
DATASET=${2:-${DATASET:-"test"}}
MODEL_CONFIG=${3:-${MODEL_CONFIG:-"/configs/jasper10x5dr_sp_offline_specaugment.yaml"}}
RESULT_DIR=${4:-${RESULT_DIR:-"./results"}}
CHECKPOINT=${5:-${CHECKPOINT:-""}}
CREATE_LOGFILE=${6:-${CREATE_LOGFILE:-"true"}}
CUDNN_BENCHMARK=${7:-${CUDNN_BENCHMARK:-"true"}}
NUM_GPUS=${8:-${NUM_GPUS:-1}}
AMP=${9:-${AMP:-"true"}}
NUM_STEPS=${10:-${NUM_STEPS:-"-1"}}
SEED=${11:-${SEED:-42}}
BATCH_SIZE=${12:-${BATCH_SIZE:-32}}
REFERENCE_FILE=${13:-${LOGITS_FILE:-"${RESULT_DIR}/${DATASET}.references"}}
PREDICTION_FILE=${14:-${PREDICTION_FILE:-"${RESULT_DIR}/${DATASET}.predictions"}}
TASK_PATH=${15:-${TASK_PATH:-"tasks.SpeechRecognition.ktelspeech.local.manifest"}}
VOCAB=${16:-${VOCAB:-"vocab"}}

mkdir -p "$RESULT_DIR"

CMD=" ../../../inference.py"
CMD+=" --task_path=$TASK_PATH"
CMD+=" --batch_size $BATCH_SIZE"
CMD+=" --dataset_dir $PWD$DATA_DIR"
CMD+=" --val_manifest $PWD$DATA_DIR/${DATASET}-wav.json"
CMD+=" --model_cfg $PWD$MODEL_CONFIG "
CMD+=" --vocab=$VOCAB"
CMD+=" --seed $SEED"
CMD+=" --ckpt $CHECKPOINT"
[ "$AMP" == "true" ] && \
CMD+=" --amp"
[ "$NUM_STEPS" -gt 0 ] && \
CMD+=" --steps $NUM_STEPS"
[ "$CUDNN_BENCHMARK" = "false" ] && \
CMD+=" --cudnn"
[ -n "$PREDICTION_FILE" ] && \
CMD+=" --save_prediction $PREDICTION_FILE"
[ -n "$REFERENCE_FILE" ] && \
CMD+=" --save_reference $REFERENCE_FILE"

if [ "$CREATE_LOGFILE" = "true" ] ; then
   export GBS=$(expr $BATCH_SIZE \* $NUM_GPUS)
   printf -v TAG "jasper_eval_benchmark_amp-%s_gbs%d" "$AMP" $GBS
   DATESTAMP=`date +'%y%m%d%H%M%S'`
   LOGFILE="${RESULT_DIR}/${TAG}.${DATESTAMP}.log"
   printf "Logs written to %s\n" "$LOGFILE"
fi

if [ "$NUM_GPUS" -gt 1  ] ; then
   CMD="python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS $CMD"
else
   CMD="python3  $CMD"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee "$LOGFILE"
fi
set +x
