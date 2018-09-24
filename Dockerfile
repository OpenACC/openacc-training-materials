# To run this dockerfil you need to present port 8888 and provide a hostname. 
# For instance:
#   $ nvidia-docker run --rm -it -p "8888:8888" -e HOSTNAME=foo.example.com openacc-labs:latest
FROM nvcr.io/hpc/pgi-compilers:ce

RUN apt update && \
    apt install -y --no-install-recommends python3-pip python3-setuptools && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir jupyter

ADD labs/ /labs
WORKDIR /labs
CMD jupyter notebook --no-browser --allow-root --ip=$HOSTNAME --port=8888
