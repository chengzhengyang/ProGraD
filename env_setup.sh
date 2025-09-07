# create environment
conda create -n prograd python=3.11.7 -y
conda activate prograd

# Install PyTorch with CUDA 12.4 support
conda install pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
