#!/bin/bash

DATA_DIR=${1:-${DATA_DIR:-"/data/KconfSpeech"}}
MODEL_CONFIG=${2:-${MODEL_CONFIG:-"/configs/jasper10x5dr_sp_offline_specaugment.yaml"}}
RESULT_DIR=${3:-${RESULT_DIR:-"./results"}}
CHECKPOINT=${4:-${CHECKPOINT:-""}}
CREATE_LOGFILE=${5:-${CREATE_LOGFILE:-"true"}}
CUDNN_BENCHMARK=${6:-${CUDNN_BENCHMARK:-"true"}}
AMP=${7:-${AMP:-"true"}}
NUM_STEPS=${8:-${NUM_STEPS:-"-1"}}
SEED=${9:-${SEED:-42}}
BATCH_SIZE=${10:-${BATCH_SIZE:-1}}
CPU=${11:-${CPU:-"false"}}
EMA=${12:-${EMA:-"false"}}
TASK_PATH=${13:-${TASK_PATH:-"tasks.SpeechRecognition.kconfspeech.local.manifest"}}
VOCAB=${14:-${VOCAB:-"vocab"}}

mkdir -p "$RESULT_DIR"

CMD="python ../../../inference.py"
CMD+=" --task_path=$TASK_PATH"
CMD+=" --batch_size $BATCH_SIZE"
CMD+=" --dataset_dir $PWD$DATA_DIR"
CMD+=" --wav $PWD$DATA_DIR/000000.wav"
CMD+=" --model_cfg $PWD$MODEL_CONFIG"
CMD+=" --vocab=$VOCAB"
CMD+=" --seed $SEED "
[ "$NUM_STEPS" -gt 0 ] && \
CMD+=" --steps $NUM_STEPS"
[ "$CUDNN_BENCHMARK" = "true" ] && \
CMD+=" --cudnn"
[ "$AMP" == "true" ] && \
CMD+=" --amp"
[ "$CPU" == "true" ] && \
CMD+=" --cpu"
[ "$EMA" == "true" ] && \
CMD+=" --ema"
[ -n "$CHECKPOINT" ] && \
CMD+=" --ckpt=${CHECKPOINT}"

if [ "$CREATE_LOGFILE" = "true" ] ; then
   export GBS=$(expr $BATCH_SIZE)
   printf -v TAG "jasper_inference_benchmark_amp-%s_gbs%d" "$AMP" $GBS
   DATESTAMP=`date +'%y%m%d%H%M%S'`
   LOGFILE="${RESULT_DIR}/${TAG}.${DATESTAMP}.log"
   printf "Logs written to %s\n" "$LOGFILE"
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
[ -n "$PREDICTION_FILE" ] && echo "PREDICTION_FILE: ${PREDICTION_FILE}"
[ -n "$LOGITS_FILE" ] && echo "LOGITS_FILE: ${LOGITS_FILE}"
