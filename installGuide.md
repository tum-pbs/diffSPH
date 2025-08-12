# Installing diffSPH

_diffSPH_ is a primarily python package that has a few ways you can install it, based on what you want to do with it. If your goal is to simply use the simulator then installing the package via _pip_ may be sufficient, but if you want to really dive into differentiable SPH schemes, you may want to install it from source. The basic requirements for _diffSPH_ are: 

- Any PyTorch Version > 2.5.0 (earliest tested)
- Any Python Version > 3.11 (earliest tested)
- Any Cuda version > 11.8 (earliest tested) also works on Mac without CUDA support

Earlier versions may work, but our preferred environment for now is PyTorch 2.7.1, Python 3.12 and CUDA 12.8. In any case, the installation process begins by setting up a virtual environment for python to create a clean starting point. If you already have an existing environment you want to install _diffSPH_ into, then  you can skip this step but your configuration might not work. If you have any problems send us an email at [contact@fluids.dev](mailto:contact@fluids.dev).

## Environment Setup

```bash
# Setup a blank virtual environment, change for the python version you want
conda create -n diffSPHEnv python=3.12 
# Activate the newly created environment
conda activate diffSPHEnv
```

Depending on what you want to do with _diffSPH_ the list of required packages may be shorter, but this set of packages allows you to run any case and any functionality:

```bash
pip install toml scipy numba tqdm h5py matplotlib ipywidgets ipympl imageio scikit-image ipykernel imageio_ffmpeg portalocker
```

For the packages and their uses:
- toml: used for parsing configuration files and setups, as well as the Domain Specific Language
- ipywidgets, ipympl, ipykernel: used for running a Jupyter environment
- scipy, scikit-image, numba: advanced numerical features and acceleration for the random noise generation
- matplotlib, imageio, imageio_ffmpeg: used for visualization
- portalocker: For batch parallelism on multiple GPUs

Once you've installed these packages and got a basic Python environment setup, you can install PyTorch:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision torchaudio
```

You may want to check if PyTorch was installed correctly by running
```bash
python -c "import torch; print(torch.__version__)"
# The output should be: 2.7.1+cu128
```

## torchCompactRadius

Now that you have the environment setup the next step is to install the C++/CUDA based neighborhood search, which is the most involved step of the entire installation procedure. Depending on your system there are four approaches for this:

1. Install a source distribution directly via pip
2. Install a precompiled package from `https://fluids.dev/torchCompactRadius/wheels/torch-{torch.__version__}/`
3. Checkout the source from Github and build a wheel for deployment
4. Checkout the source from Github and intall locally

If you want to install the neighborhood search in any way that requires compilation, you may want to set the `TORCH_CUDA_ARCH_LIST` environment variable to limit compilation, e.g., `TORCH_CUDA_ARCH_LIST = '8.0;8.6'`, as otherwise the compilation script will build the library for all visible architectures on your system, which can take a long time.

## Install via pip

The first approach is the easiest and should work on any system and can be executed by simply calling

```bash
pip install torchCompactRadius
```

This will checkout the source from PyPI and then build the library locally. This does require GCC and CUDA, but with the environment setup as before, this is done already. One thing to note here is that the compilation process is silent and displays very limited output. The only sign of progress is the small progress wheel turning once in a while. (The compilation should take around 5 minutes.)


## Install a precompiled package

For some versions of PyTorch and CUDA we have precompiled the neighborhood search so you can install it directly, without needing to build it locally. This can be done by first getting  the PyTorch version as above:

```bash
python -c "import torch; print(torch.__version__)"
```

And then putting the output directly into
```bash
pip install torchCompactRadius --index_url https://fluids.dev/torchCompactRadius/wheels/torch-{torch.__version__}/ 
```

This installation should be much faster but there may not be a precompiled binary for your specific setup available.

## Installation from source

The most flexible way of installing the neighborhood search is to directly build it from source. For this you should first checkout the source

```bash
git clone https://github.com/wi-re/torchCompactRadius
cd torchCompactRadius
```

And then you can build it as the environment is already setup with support for this step. To build it you can either call

```bash
python -m build 
```

or

```bash
python setup.py install
```

The former will compile the library and produce a wheel you can deploy, e.g., to a remote system, and install afterwards, whereas the latter option will install the library only locally. You can also use `develop` for `setup.py`, but this is not recommended as this will only use on-the-fly changes of the python part, not the CUDA part, which can lead to difficult to trace bugs.

## Next Steps:

Once you are done installing torchCompactRadius, verify that it installed correctly by calling
```bash
python -c "import torchCompactRadius"
```

Now all that is left is to install _diffSPH_, which you can either do directly via pip:
```bash
pip install diffSPH
```

Or from source
```bash
git clone https://github.com/tum-pbs/diffSPH
cd diffSPH
pip install -e .
```

You are now ready to start using our differentiable SPH solver. In short the commands below will get you going from scratch:
```bash
conda create -n diffSPHEnv python=3.12
conda activate diffSPHEnv
conda install nvidia/label/cuda-12.8.1::cuda-toolkit
pip install toml scipy numba tqdm h5py matplotlib ipywidgets ipympl imageio scikit-image ipykernel imageio_ffmpeg portalocker
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision torchaudio
pip install torchCompactRadius
pip install diffSPH
```