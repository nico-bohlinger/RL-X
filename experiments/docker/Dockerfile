FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG MAMBA_ENV=rlx
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends git hwloc curl ca-certificates bzip2 openssh-client && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && chmod +x /usr/local/bin/micromamba && micromamba --version
RUN micromamba create -y -n ${MAMBA_ENV} python=3.11.4 pip && micromamba clean -a -y

WORKDIR /RL-X_ws
RUN git clone https://github.com/nico-bohlinger/RL-X.git
WORKDIR /RL-X_ws/RL-X
RUN micromamba run -n ${MAMBA_ENV} python -m pip install pip -U
RUN micromamba run -n ${MAMBA_ENV} pip install -e .[all]
RUN micromamba run -n ${MAMBA_ENV} pip uninstall $(micromamba run -n ${MAMBA_ENV} pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
RUN micromamba run -n ${MAMBA_ENV} pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu118 --upgrade
RUN micromamba run -n ${MAMBA_ENV} pip install "jax[cuda12]"

RUN mkdir /root/.ssh/
ADD for-docker-without-pw /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]