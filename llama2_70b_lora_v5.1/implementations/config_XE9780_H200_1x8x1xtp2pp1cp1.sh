#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source $(dirname ${BASH_SOURCE[0]})/config_XE9780_common.sh

# hyperparameters
export NCCL_TEST=0
export LR=0.0005
export MAX_STEPS=896
export MINIBS=1
export TP=2
export PP=1
export CP=1
export SP=1
export TP_COMM_OVERLAP=1 
export VBOOST_VALUE=1 
export GPU_ARCH="h"

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=True
export FP8_AMAX_HISTORY=32

export SKIP_EVALS=3
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=1
export SHARP=True
export WALLTIME_RUNANDTIME=500
export WALLTIME=$((5 + ${NEXP:-10} * ($WALLTIME_RUNANDTIME + 5)))
export MLPERF_NUM_NODES=$DGXNNODES

# GMO
export DGXSYSTEM="GMO_GPU_CLOUD"
export DGXNGPU=8
export PMIX_MCA_gds=^ds12
export work_dir="/path/to/mlperf_training/llama2_70b_lora_v5.1/implementations"
export DATADIR=$work_dir/data/gov_report
export MODEL=$work_dir/data/model
export LOGDIR=$work_dir/logs
export CONT=""
export SLURM_MPI_TYPE=pmix

# NCCL
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NVTE_UB_SOCKET_IFNAME=bond0.1505
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_7,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SL=1
export NCCL_IB_ADAPTIVE_ROUTING=1
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc
