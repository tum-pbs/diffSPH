conda create --name sphEnv python=3.12
conda activate sphEnv
conda install -c anaconda ipykernel -y
conda install nvidia/label/cuda-12.8.1::cuda-toolkit cudnn
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install toml scipy numba tqdm h5py matplotlib ipywidgets ipympl imageio scikit-image