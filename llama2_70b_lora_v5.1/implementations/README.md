## Running NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 300GB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPU is not needed for preprocessing scripts, but is needed for training.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
$ docker build --network=host -t <docker/registry>/mlperf-nvidia:<tag> .
...
$ docker push <docker/registry>/mlperf-nvidia:<tag>
```

### 3.2 Download dataset and model

This benchmark uses the [GovReport](https://gov-report-data.github.io/) dataset.  
You can reuse the procedure for MLPerf LLaMA2 70B v4.1. Please refer to [this steps](../../llama2_70b_lora_v4.1/implementations#32-download-dataset-and-model).

### 3.3 Preprocess dataset and model
You also can reuse the procedure for MLPerf LLaMA2 70B v4.1. Please refer to [this steps](../../llama2_70b_lora_v4.1/implementations#33-preprocess-dataset-and-model).

After conversion you should see the following files in the `/data` directory:
```bash
gov_report/
    train.npy
    validation.npy
model/
    <hash>_tokenizer.model
    llama2-70b.nemo
    model_config.yaml
    model_weights
```

Exit the container.

## 4. Launch training

### 4.1 Setup environment value
Configure the following values according to your environment.
* `config_XE9780_common.sh`
```bash
export WORK_DIR="/path/to/mlperf_training/llama2_70b_lora_v5.1/implementation" <<< path/to
```
* `config_XE9780_H200_1x8x1xtp2pp1cp1.sh`
```bash
export work_dir="/path/to/mlperf_training/llama2_70b_lora_v5.1/implementations" <<< path/to
export CONT="" <<< <docker/registry>/mlperf-nvidia:<tag>
```

### 4.2 Launch the training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
$ source config_XE9780_H200_1x8x1xtp2pp1cp1.sh  # select config and source it
$ sbatch -p <PARTITION> -N $DGXNNODES -t $WALLTIME --gpus-per-node $DGXNGPU run.sub  # you may be required to set --account and --partition here
```

## 5. Evaluation

### Quality metric
Cross entropy loss

### Quality target
0.925

### Evaluation frequency
Every 384 sequences, CEIL(384 / global_batch_size) steps if 384 is not divisible by GBS. Skipping first FLOOR(0.125*global_batch_size+2) evaluations

### Evaluation thoroughness
Evaluation on the validation subset that consists of 173 examples
