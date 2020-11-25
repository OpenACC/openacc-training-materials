# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

# To build the docker container, run: $ sudo docker build -t openacc-labs:latest .
# To run: $ sudo docker run --rm -it --runtime nvidia -p 8888:8888 openacc-labs:latest
# Finally, open http://localhost:8888/

FROM nvcr.io/nvidia/nvhpc:20.9-devel-ubuntu20.04

RUN apt-get -y update && \
        DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends python3-pip python3-setuptools nginx zip make build-essential && \
        rm -rf /var/lib/apt/lists/* && \
        pip3 install --no-cache-dir jupyter &&\
        mkdir -p /home/openacc/labs

############################################
# NVIDIA nsight-systems-2020.4.1 
RUN apt-get update -y && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget && \
        apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys F60F4B3D7FA2AF80 && \
        echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list &&\
        apt-get update -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2020.4.1

#################################################
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/lib64/"
ENV PATH="/opt/nvidia/nsight-systems/2020.4.1/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/include:$PATH"
#################################################

ADD docker-configs/default /etc/nginx/sites-available/default

ADD labs/ /home/openacc/labs/
WORKDIR /home/openacc/labs
CMD service nginx start && jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/home/openacc/labs