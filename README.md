# Gaussian Compression


## Requirements

- An __NVIDIA GPU__ with __CUDA__ support
- A __C++14__ capable compiler. The following choices are recommended and have been tested:
    - __Windows:__ Visual Studio 2019 or 2022
    - __Linux:__ GCC/G++ 8 or higher
- A recent version of __[CUDA](https://developer.nvidia.com/cuda-toolkit)__. The following choices are recommended and have been tested:
    - __Windows:__ CUDA 11.8 or higher
    - __Linux:__ CUDA 11.8 or higher
- __[CMake](https://cmake.org/) v3.21 or higher__.

Install [CUDA](https://developer.nvidia.com/cuda-toolkit) in `/usr/local/` and add the CUDA installation to your PATH.
For example, if you have CUDA 11.8, add the following to your `~/.bashrc`
```sh
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH # WSL users only
```

WSL users only need to copy the `lib` folder from `/usr/lib/wsl/` to `/usr/local/cuda-11.8/lib64/`:
```sh
sudo cp -r /usr/lib/wsl/lib/* /usr/local/cuda-11.8/lib64/
```

## Prerequisites

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:AndreiCod/gaussian-compression.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/AndreiCod/gaussian-compression.git --recursive
```



## Installation

```shell
conda env create -f environment.yml
conda activate gaussian_compression
```

## Dataset and Model Preparation

Create a `datasets` folder in the root directory and put the COLMAP dataset in it. The dataset should have the following structure:
```
datasets
└── dataset_name
    ├── images
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    ├── sparse
```

Create a `input` folder in the root directory and put the gaussian model in it. The model should have the following structure:
```
input
└── model_name
    ├── point_cloud
    ├──train
    ├──cameras.json
    ├──cfg_args
    ├──input.ply
    ├──per_view.json
    ├──results.json

```

`dataset_name` and `model_name` should be the same.

## Training

Modify the `gaussian_model_path` inside the `compression_config.yml.default` accordingly and save it as `compression_config.yml`.

```shell
python train.py --config compression_config.yml
```

If the training is successful, the model will be saved in a `compression` folder inside the `gaussian_model_path`.


## Rendering
In order to render the compressed model, run the following command:

```shell
python render.py --model_path gaussian_model_path
```

## Evaluation
If there is a test folder in the `gaussian_model_path`, you can evaluate the model by running the following command:

```shell
python metrics.py --model_path gaussian_model_path/compression/iteration_number
```

The evaluation results will be saved in the `gaussian_model_path/compression/iteration_number` folder.

## Batch evaluation
If you have multiple models to evaluate, you can run the following command that will train, render and evaluate:

```shell
python batch_eval.py --config compression_config.yml --models_dir input
```


## Viewers
The compression model also exports a point cloud. You can visualize it using the SIBR viewer. You can find the viewer [here](https://sibr.gitlabpages.inria.fr/?page=index.html&version=0.9.6).