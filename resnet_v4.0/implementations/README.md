# 1. Requirements
* [MXNet 24.04-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet)
* [GMO GPU Cloud](https://gpucloud.gmo/)

# 2. Directions
## Steps to download and verify data
1.  Clone the public DeepLearningExamples repository and Build the container
```bash
$ cd $HOME && git clone https://github.com/NVIDIA/DeepLearningExamples
$ cd DeepLearningExamples/MxNet/Classification/RN50v1.5
$ git checkout 81ee705868a11d6fe18c12d237abe4a08aab5fd6
```
2. Login to Compute Node and setup environment
```bash
$ salloc -p <PARTITION NAME> -N 1 -G 8
$ ssh <NODENAME>
$ export work_dir="$HOME/DeepLearningExamples/MxNet/Classification/RN50v1.5" && cd $work_dir
$ mkdir train val data
```
3. Download and unpack the data for training on Compute Node
* downloaded data is approximately 138 GB and will take several hours to download and deploy
```bash
$ cd $work_dir/train && wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
$ tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
$ find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
```
4. Download and unpack the data for validation on Compute Node
* This process is relatively quick
```bash
$ cd $work_dir/val && wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
$ tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
$ wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
5. Exit from the node and terminate the job
```bash
$ exit
$ exit
```
## Steps to preprocess the dataset on Login Node
```bash
$ export work_dir="$HOME/DeepLearningExamples/MxNet/Classification/RN50v1.5"
$ cd $work_dir
$ module load singularitypro
$ srun -p part-cpu singularity pull nvidia_rn50_mx.sif docker://<REPO NAME>/nvidia_rn50_mx:latest
$ srun -p part-group_xxxxxx -G 8 singularity exec --nv -B $work_dir/data:/data nvidia_rn50_mx.sif ./scripts/prepare_imagenet.sh $work_dir /data
         ^^^^^^^^^^^^^^^^^^ GPU Partition
```
#### NOTE
Containers must have been built according to the [official procedure](https://github.com/mlcommons/training_results_v4.0/tree/main/NVIDIA/benchmarks/resnet/implementations/h200_ngc24.04_mxnet#steps-to-download-and-verify-data) and pushed to your repositories in advance
```bash
$ git clone https://github.com/NVIDIA/DeepLearningExamples
$ cd DeepLearningExamples/MxNet/Classification/RN50v1.5
$ git checkout 81ee705868a11d6fe18c12d237abe4a08aab5fd6
$ docker build . -t <REPO NAME>/nvidia_rn50_mx:latest
...
$ docker push <REPO NAME>/nvidia_rn50_mx:latest
```

## Steps to run benchmark
1. Prepare the container
* **This must be done in a docker-enabled bastion**
```bash
$ cd $HOME && git clone https://github.com/masamokkulu/mlperf_training.git
$ export work_dir="$HOME/mlperf_training/resnet_v4.0/implementations" && cd $work_dir
$ docker build . -t <REPO NAME>/mlperf-nvidia:nvidia_rn50_mx
...
$ docker push <REPO NAME>/mlperf-nvidia:nvidia_rn50_mx
```
2. Run benchmark
* On GMO GPU Cloud
```bash
$ cd $HOME && git clone https://github.com/masamokkulu/mlperf_training.git
$ export work_dir="$HOME/mlperf_training/resnet_v4.0/implementations" && cd $work_dir
$ srun -p part-cpu singularity pull mlperf-nvidia.sif docker://<REPO NAME>/mlperf-nvidia:nvidia_rn50_mx
$ source config_DGXH100.sh
$ sbatch -p <PARTITION NAME> -G 8 -N $DGXNNODES run.sub
```
