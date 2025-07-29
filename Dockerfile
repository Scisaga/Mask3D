FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG PYTHON_VERSION=3.10.9

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PROJECT_DIR=/opt/mask3d

# 安装常用工具、ssh、python编译依赖等
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl ca-certificates unzip git vim htop sudo openssh-server \
    zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev tk-dev libgdbm-dev libnss3-dev liblzma-dev uuid-dev && \
    rm -rf /var/lib/apt/lists/*

# 编译安装 Python
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    tar -xJf Python-${PYTHON_VERSION}.tar.xz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --prefix=/opt/python-${PYTHON_VERSION} --enable-optimizations --with-lto && \
    make -j"$(nproc)" && make install && \
    ln -s /opt/python-${PYTHON_VERSION}/bin/python3 /usr/local/bin/python3 && \
    ln -s /opt/python-${PYTHON_VERSION}/bin/pip3 /usr/local/bin/pip3 && \
    cd / && rm -rf /tmp/Python-${PYTHON_VERSION}*

# 安装 Conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/conda.sh && \
    bash /tmp/conda.sh -b -p /opt/conda && rm /tmp/conda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" >> /etc/profile.d/conda.sh
ENV PATH="/opt/conda/bin:$PATH"

# 安装 PyTorch for CUDA 11.3
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
      --extra-index-url https://download.pytorch.org/whl/cu113

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

# 拷贝 Mask3D 项目到指定目录（重点）
COPY . ${PROJECT_DIR}
RUN chown -R dev:dev ${PROJECT_DIR}

# 切换默认目录和用户
USER dev
WORKDIR ${PROJECT_DIR}

EXPOSE 22 5000
CMD ["/usr/sbin/sshd", "-D", "-e"]
