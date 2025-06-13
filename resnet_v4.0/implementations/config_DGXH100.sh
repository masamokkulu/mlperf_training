source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

## DL params
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="400"
export KVSTORE="horovod"
export LR="11.0"
export WARMUP_EPOCHS="2"
export EVAL_OFFSET="2" # Targeting epoch no. 35
export EVAL_PERIOD="4"
export WD="5.0e-05"
export MOM="0.9"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS=${NUMEPOCHS:-"37"}

export NETWORK="resnet-v1b-mainloop-fl"
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_BWD=0
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_FWD=0

export DALI_THREADS="6"
export DALI_PREFETCH_QUEUE="5"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_HW_DECODER_LOAD="0.99"
export DALI_CACHE_SIZE="12288"
export DALI_ROI_DECODE="1"
export DALI_DONT_USE_MMAP=1
export MXNET_GPU_WORKER_NSTREAMS=1 

#DALI buffer presizing hints
export DALI_PREALLOCATE_WIDTH="5980"
export DALI_PREALLOCATE_HEIGHT="6430"
export DALI_DECODER_BUFFER_HINT="1315942" #1196311*1.1
export DALI_CROP_BUFFER_HINT="165581" #150528*1.1
export DALI_TMP_BUFFER_HINT="355568328" #871491*batch_size
export DALI_NORMALIZE_BUFFER_HINT="441549" #401408*1.1

export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=2
export MXNET_HOROVOD_NUM_GROUPS=1
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999
export MXNET_EXTENDED_NORMCONV_SUPPORT=1


## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=20
if [[ ${MLPERF_POWER_TRAIN_AFTER_RUN_STOP} -gt 0 ]]; then
  WALLTIME_MINUTES=$((${WALLTIME_MINUTES} + 15))
  export SUSTAINED_TRAINING_TIME=10
fi
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]] || [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=1230
  export MINEDP_CLK=1620
fi
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))
export CONT=$work_dir/mlperf-nvidia.sif
export DATADIR=$HOME/DeepLearningExamples/MxNet/Classification/RN50v1.5/data
export MLPERF_CLUSTER_NAME="GMO_GPU_GLOUD"
export PMIX_MCA_gds=^ds12
export SINGULARITYENV_PYTHONUSERBASE=/usr/local/lib/python3.10/dist-packages
export NCCL_TEST=0
