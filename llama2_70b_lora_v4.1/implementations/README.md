## Running NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark
This file contains the instructions for running the NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements
- At least 300GB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPU is not needed for preprocessing scripts, but is needed for training.

## 2. Software Requirements
- [GMO GPU Cloud](https://gpucloud.gmo/)

## 3. Set up
### 3.1 Build the container
Replace `<docker/registry>` with your container registry and build:
```bash
$ git clone https://github.com/masamokkulu/mlperf_training.git && cd mlperf_training/llama2_70b_lora_v4.1/implementations
$ docker build -t <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
...
$ docker push <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
```
* Please do so in a Docker-enabled step-up environment

### 3.2 Download dataset and model
This benchmark uses the [GovReport](https://gov-report-data.github.io/) dataset.
First, prepare the singularity container to be used for verification.
```bash
$ cd $HOME && git clone https://github.com/masamokkulu/mlperf_training.git
$ cd mlperf_training/llama2_70b_lora_v4.1/implementations
$ export work_dir=$(pwd)
$ module load singularitypro && srun -p <PARTITION NAME> singularity pull mlperf-nvidia.sif docker://<docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
...
```
### 3.3 Preprocess dataset and model
Continue with the previous docker container running and convert dataset to numpy format:
```bash
$ mkdir data
$ srun -p <PARTITION NAME> -G 8 singularity exec --nv \
  -B $work_dir/data:/data \
  mlperf-nvidia.sif \
  python scripts/download_dataset.py --data_dir /data/gov_report
...
$ srun -p <PARTITION NAME> -G 8 singularity exec --nv \
  -B $work_dir/data:/data \
  mlperf-nvidia.sif \
  python scripts/download_model.py --model_dir /data/model
...
```
then, continue with the conversion of the dataset and model:
* NOTE: `/scratch` must be large enough (TB ~)
```bash
$ srun -p <PARTITION NAME> -G 8 singularity exec --nv \
  -B $work_dir/data:/data \
  mlperf-nvidia.sif \
  python scripts/convert_dataset.py --data_dir /data/gov_report
...
$ srun -p <PARTITION NAME> -G 8 singularity exec --nv \
  -B $work_dir/data:/data \
  -B /scratch/dir1:/tmp \
  -B /scratch/dir2:/var/tmp \
  mlperf-nvidia.sif \
  python scripts/convert_model.py \
  --input_name_or_path=/data/model \
  --output_path=/data/model/llama2-70b.nemo
...

$ cd data/model && srun -p part-cpu find . -type f ! -name 'llama2-70b.nemo' -exec rm -f {} + && tar -xvf llama2-70b.nemo
```
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
## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
$ cd $work_dir && mkdir logs
$ source config_DGXH200_1x8x2xtp1pp1cp2.sh
$ sbatch -p <PARTITION NAME> -N $DGXNNODES  -G 8 run.sub
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

## Known Issue
#### `AssertionError: global batch size (8) is not divisible by micro batch size (4) times data parallel size (8)` @ `convert_model.py`
Error due to global_batchsize not divisible by micro_batch*pararell. Please modify [megatron_llama_config.yaml](./scripts/megatron_llama_config.yaml) to adjust batch size.
```
model:
  mcore_gpt: True
  # specify micro_batch_size, global_batch_size, and model parallelism
  # gradient accumulation will be done automatically based on data_parallel_size
  micro_batch_size: 4 # limited by GPU memory 
  global_batch_size: 8 # will use more micro batches to reach global batch size
                     ^ increase 32 from 8
```


