FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PROJECT_DIR=/opt/mask3d

# 安装常用工具、ssh、编译依赖、openblas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl ca-certificates unzip git vim htop sudo openssh-server \
    zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev tk-dev libgdbm-dev libnss3-dev liblzma-dev uuid-dev \
    libopenblas-dev cmake make && \
    rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/conda.sh && \
    bash /tmp/conda.sh -b -p /opt/conda && rm /tmp/conda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" >> /etc/profile.d/conda.sh
ENV PATH="/opt/conda/bin:$PATH"

# 拷贝环境配置和项目代码
COPY environment.yml ${PROJECT_DIR}/environment.yml
COPY . ${PROJECT_DIR}
RUN chown -R root:root ${PROJECT_DIR}

# 创建 conda 环境并安装依赖
WORKDIR ${PROJECT_DIR}
RUN conda env create -f environment.yml && conda clean -afy
ENV CONDA_DEFAULT_ENV=mask3d_cuda113
ENV PATH="/opt/conda/envs/mask3d_cuda113/bin:$PATH"
SHELL ["/bin/bash", "-c"]

# 安装 PyTorch、torchvision、torch-scatter、detectron2、pytorch-lightning
RUN source activate mask3d_cuda113 && \
    pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html && \
    pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps && \
    pip install --no-cache-dir pytorch-lightning==1.7.2

# 编译 third_party 依赖
RUN source activate mask3d_cuda113 && \
    mkdir -p third_party && cd third_party && \
    git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" && \
    cd MinkowskiEngine && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    python setup.py install --force_cuda --blas=openblas && \
    cd .. && \
    git clone https://github.com/ScanNet/ScanNet.git && \
    cd ScanNet/Segmentator && git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && make && \
    cd ../../pointnet2 && python setup.py install

# 创建用户 dev
RUN useradd -m -s /bin/bash -u 1000 dev && \
    mkdir -p /var/run/sshd /home/dev/.ssh && \
    chmod 700 /home/dev/.ssh && chown -R dev:dev /home/dev/.ssh && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev && \
    chmod 0440 /etc/sudoers.d/dev

# 拷贝 SSH 配置和密钥
COPY sshd_config /etc/ssh/sshd_config
COPY authorized_keys /home/dev/.ssh/authorized_keys
RUN chown dev:dev /home/dev/.ssh/authorized_keys && chmod 600 /home/dev/.ssh/authorized_keys

# 切换默认目录和用户
USER dev
WORKDIR ${PROJECT_DIR}

EXPOSE 22 5000
CMD ["/usr/sbin/sshd", "-D", "-e"]
