# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

# To build the docker container, run: $ sudo docker build -t openacc-labs:latest .
# To run: $ sudo docker run --rm -it --runtime nvidia -p 8888:8888 openacc-labs:latest
# Finally, open http://localhost:8888/
ARG NVHPC_ARCH="Linux_x86_64"

FROM nvcr.io/nvidia/nvhpc:24.7-devel-cuda12.5-ubuntu24.04

WORKDIR /home/openacc/labs

RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends python3-pip python3-setuptools nginx zip make build-essential && \
    rm -rf /var/lib/apt/lists/* 

ADD requirements.txt /home/openacc/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /home/openacc/requirements.txt 

#################################################
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib:/opt/nvidia/hpc_sdk/${NVHPC_ARCH}/2024/compilers/lib"
ENV PATH="/opt/nvidia/hpc_sdk/${NVHPC_ARCH}/2024/compilers/bin:/opt/nvidia/hpc_sdk/${NVHPC_ARCH}/2024/comm_libs/mpi/bin:/opt/nvidia/hpc_sdk/${NVHPC_ARCH}/2024/profilers/Nsight_Systems/bin:/opt/nvidia/hpc_sdk/${NVHPC_ARCH}/2024/profilers/Nsight_Compute:$PATH"
#################################################

ADD docker-configs/default /etc/nginx/sites-available/default

ADD labs/ /home/openacc/labs/
CMD service nginx start && jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/home/openacc/labs --JupyterNotebookApp.default_url="/tree/README.ipynb"