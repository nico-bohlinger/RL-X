FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git
RUN apt-get install -y python3-pip

WORKDIR /RL-X_ws
RUN git clone https://github.com/nico-bohlinger/RL-X.git
WORKDIR /RL-X_ws/RL-X
RUN python3 -m pip install pip -U
RUN pip3 install -e .[all]
RUN pip3 install "torch==2.1.2" --index-url https://download.pytorch.org/whl/cu118 --upgrade
RUN pip3 uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
RUN pip3 install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]