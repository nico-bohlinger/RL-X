# Detailed Installation Guide
### 1. Conda
For Linux, MacOS and Windows, a conda environment is recommended.  
All the code was tested with Python 3.11.4, other versions might work as well.
```
conda create -n rlx python=3.11.4
conda activate rlx
```

### 2. RL-X
For Linux, MacOS and Windows, RL-X has to be cloned.
```
git clone git@github.com:nico-bohlinger/RL-X.git
cd RL-X
```

### 3. Dependencies
For Linux, all dependencies can be installed with the following command:
```
pip install -e .[all]
```
For MacOS and Windows, EnvPool is currently not supported. Therefore, the following command has to be used:
```
pip install -e .
```

### 4. PyTorch
For Linux, MacOS and Windows, PyTorch has to be installed separately to use the CUDA 11.8 version such that there are no conflicts with JAX.
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 5. JAX
For Linux, JAX with GPU support can be installed with the following command:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For MacOS and Windows, JAX with GPU support is not supported out-of-the-box. However, it can be done with some extra effort (see [here](https://github.com/google/jax) for more information).
