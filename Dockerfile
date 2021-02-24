FROM ubuntu:20.04
SHELL ["/bin/bash", "--login", "-c"]

ENV TZ=Europe/Berlin

# To solve invoke-rc.d error for openssh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
&& printf '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d \
&& apt-get update \
&& apt-get install -y ssh \
    gcc-9 \
    g++-9 \
    cmake \
    git \
    wget \
    tmux \
    clang \
    rsync \
    tar \
    dirmngr\
&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
&&  apt-get install --no-install-recommends -y \
    software-properties-common \
&& apt-get clean

RUN wget \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin \
    && mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" \
    && apt-get update \
    && apt-get -y --no-install-recommends install cuda-toolkit-10-2 libnvidia-ml-dev \
    && apt-get clean

RUN useradd -m user \
  && yes password | passwd user

USER user

WORKDIR /home/user/
COPY --chown=user ./environment.yml .
ENV PATH="/home/user/miniconda3/bin:${PATH}"
ARG PATH="/home/user/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /home/user/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash \
    && conda env create -n pytorch --file environment.yml \
    && conda clean -a -y \
    && chmod -R a+rw /home/user

USER root
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin no'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_clion

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_clion"]

