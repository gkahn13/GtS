FROM nvidia/cudagl:9.0-devel-ubuntu16.04

ARG UID=1000
ARG GID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gcc \
        libc6-dev \
        libglu1 \
        libglu1:i386 \
        libsm6 \
        libxv1 \
        libxv1:i386 \
        make \
        python \
        python-numpy \
        x11-xkb-utils \
        xauth \
        xfonts-base \
        xkb-data && \
    apt-get install --reinstall -y build-essential && \
    apt-get install -y \
        sudo \
        nano \
        wget \
        bzip2 \
        gcc \
        g++ \
        git \
        tmux && \
    rm -rf /var/lib/apt/lists/*


ENV DISPLAY :0




RUN groupadd -g	$GID gcg-user
RUN useradd -m -u $UID -g $GID gcg-user && echo "gcg-user:gcg" | chpasswd && adduser gcg-user sudo
USER gcg-user

ENV HOME /home/gcg-user
WORKDIR $HOME

ENV SOURCEDIR $HOME/source
RUN mkdir $SOURCEDIR

# install miniconda
RUN cd $SOURCEDIR && \   
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    bash Miniconda3-4.5.4-Linux-x86_64.sh -b -p $SOURCEDIR/miniconda && \
    rm Miniconda3-4.5.4-Linux-x86_64.sh && \
    cd
ENV PATH $SOURCEDIR/miniconda/envs/gcg/bin:$SOURCEDIR/miniconda/bin:$PATH
# setup gcg miniconda env
RUN conda create -y -n gcg python=3.5 && \
    echo 'source activate gcg' >> ~/.bashrc
# install to gcg env
RUN conda install -n gcg -y cudnn
RUN pip install tensorflow-gpu==1.8.0
RUN pip install panda3d==1.10.0.dev1182
RUN pip install colorlog==3.1.0
RUN pip install pandas==0.21.0
RUN conda install -n gcg -y pillow=5.0.0
RUN conda install -n gcg -y matplotlib=2.2.2
RUN pip install ipython==6.4.0

# setup gcg
RUN echo 'export PYTHONPATH=$PYTHONPATH:$HOME/gcg/src' >> ~/.bashrc

# setup tmux
RUN echo 'set-option -g default-shell /bin/bash' >> ~/.tmux.conf

# setup gibson
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 
RUN pip install torchvision==0.2.0

RUN echo 'gcg' | sudo -S apt-get update
RUN echo 'gcg' | sudo -S apt-get install -y git build-essential cmake libopenmpi-dev 		
RUN echo 'gcg' | sudo -S apt-get install -y zlib1g-dev tmux
RUN echo 'gcg' | sudo -S apt-get install -y \
		libglew-dev \
		libglm-dev \
		libassimp-dev \
		xorg-dev \
		libglu1-mesa-dev \
		libboost-dev \
		mesa-common-dev \
		freeglut3-dev \
		libopenmpi-dev \
		cmake \
		golang \
		libjpeg-turbo8-dev \
		wmctrl \ 
		xdotool
RUN echo 'gcg' | sudo -S apt-get install -y vim wget unzip 
RUN echo 'gcg' | sudo -S apt-get install -y libzmq3-dev
RUN echo 'gcg' | sudo -S apt-get clean && \
	echo 'gcg' | sudo -S apt-get autoremove && \
    echo 'gcg' | sudo -S rm -rf /var/lib/apt/lists/* && \
	echo 'gcg' | sudo -S rm -rf /var/cache/apk/*


RUN cd $SOURCEDIR && \
    wget https://people.eecs.berkeley.edu/~gregoryk/downloads/GibsonEnvGtS.tar.gz
RUN cd $SOURCEDIR && \
    rm -rf GibsonEnv && \
    tar -xvf GibsonEnvGtS.tar.gz && \
    rm GibsonEnvGtS.tar.gz && \
    cd GibsonEnv && \
    bash build.sh build_local && \
    pip install -e . && \
    cd

RUN pip install scipy==1.1.0

