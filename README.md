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