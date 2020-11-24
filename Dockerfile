# To run this dockerfile you need to present port 8000 and provide a hostname.
# For instance:
#   $ docker run --runtime nvidia --rm -it -p "8000:8000" -e HOSTNAME=foo.example.com openacc-labs:latest

FROM nvcr.io/nvidia/nvhpc:20.9-devel-ubuntu20.04

# PGI Tutorials

ARG TURBOVNC_VERSION=2.2.1
ARG VIRTUALGL_VERSION=2.6.1
ARG LIBJPEG_VERSION=1.5.2
ARG WEBSOCKIFY_VERSION=0.8.0
ARG NOVNC_VERSION=1.0.0
ARG VNCPASSWORD=openacc

RUN dpkg --add-architecture i386 && \
        apt update && \
        apt install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        nginx \
        zip \
        xfce4 \
        dbus-x11 \
        ca-certificates \
        curl 

RUN     apt update -y && \
        apt upgrade -y && \
        apt install -y --no-install-recommends \
        libc6-dev \
        libglu1 libglu1:i386 \
        libsm6 \
        libxv1 libxv1:i386 \
        make \
        python \
        python-numpy \
        x11-xkb-utils \
        xauth \
        xfonts-base \
        xkb-data \
        xfce4-terminal \
        openjdk-8-jdk \
        # libxau-dev \
        # libxdmcp-dev \
        # libxcb1-dev \
        pkg-config\
        #ubuntu 20.04
        libglvnd0 libglvnd0:i386 \ 
        libgl1 libgl1:i386 \ 
        libglx0 libglx0:i386 \ 
        libegl1 libegl1:i386 \
        libgles2 libgles2:i386 &&\
        # libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev\  
        # libxext6 libx11-6\
        ## may not be needed #ibxext-dev:i386  libx11-dev:i386 
        # libxext-dev \ 
        #libx11-dev  \
        # libxkbcommon-x11-0 \
        # libxkbcommon-x11-dev\
        #libgl1-mesa-dri libgl1-mesa-glx \
        #x11proto-gl-dev &&\
        #libqt4-declarative qt4-dev-tools qt4-qmake libqtwebkit4 && \
        pip3 install --no-cache-dir jupyter && \
        useradd -k /etc/skel -m -s /usr/local/bin/entrypoint.sh -p openacc openacc && \
        echo 'openacc:openacc' | chpasswd && \
        rm -rf /var/lib/apt/lists/* && \
        mkdir /var/run/sshd && \
        mkdir /home/openacc/labs

# Install missing dependency or do ln -s /lib/x86_64-linux-gnu/libtinfo.so.6 /lib/x86_64-linux-gnu/libtinfo.so.5 
RUN apt-get update && apt-get install -y libtinfo5 
#RUN apt-get install -y build-essential qt5-default qtcreator

COPY --from=nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04  \
        /usr/lib/x86_64-linux-gnu \
        /usr/lib/x86_64-linux-gnu

COPY --from=nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04 \
        /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN \
        echo '/usr/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
        echo '/usr/lib/i386-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
        ldconfig && \
        echo '/usr/${LIB}/libGL.so.1' >> /etc/ld.so.preload && \
        echo '/usr/${LIB}/libEGL.so.1' >> /etc/ld.so.preload 

# creating symlink for the driver
RUN mkdir /usr/lib/dri/ && ln -s /usr/lib/i386-linux-gnu/dri/swrast_dri.so /usr/lib/dri/swrast_dri.so
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
        #echo "deb https://developer.download.nvidia.com/devtools/repo-deb/x86_64/ /" >> /etc/apt/sources.list.d/nsight.list &&\
        apt-get update -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2020.4.1

RUN apt-get install --no-install-recommends -y build-essential

############################################
# Configure desktop
ENV DISPLAY :1
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display

# Install TurboVNC
RUN cd /tmp && \
        curl -fsSL -O -k https://svwh.dl.sourceforge.net/project/turbovnc/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb \
        -O https://svwh.dl.sourceforge.net/project/libjpeg-turbo/${LIBJPEG_VERSION}/libjpeg-turbo-official_${LIBJPEG_VERSION}_amd64.deb \
        -O https://svwh.dl.sourceforge.net/project/virtualgl/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb \
        -O https://svwh.dl.sourceforge.net/project/virtualgl/${VIRTUALGL_VERSION}/virtualgl32_${VIRTUALGL_VERSION}_amd64.deb && \
        dpkg -i *.deb && \
        rm -f /tmp/*.deb && \
        sed -i 's/$host:/unix:/g' /opt/TurboVNC/bin/vncserver
ENV PATH ${PATH}:/opt/VirtualGL/bin:/opt/TurboVNC/bin

# Install NoVNC
RUN curl -fsSL https://github.com/novnc/noVNC/archive/v${NOVNC_VERSION}.tar.gz | tar -xzf - -C /opt && \
        curl -fsSL https://github.com/novnc/websockify/archive/v${WEBSOCKIFY_VERSION}.tar.gz | tar -xzf - -C /opt && \
        mv /opt/noVNC-${NOVNC_VERSION} /opt/noVNC && \
        mv /opt/websockify-${WEBSOCKIFY_VERSION} /opt/websockify && \
        cd /opt/websockify && make
# Insecure by default. TODO: randomize?
RUN mkdir -p /home/openacc/.vnc && echo "$VNCPASSWORD" | vncpasswd -f > /home/openacc/.vnc/passwd && chmod 0600 /home/openacc/.vnc/passwd

# Overlay file system with overrides
COPY fs /

# Default panel (otherwise prompted to initialize panel)
RUN cp /etc/xdg/xfce4/panel/default.xml /etc/xdg/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml

#################################################33
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/lib64/"
ENV PATH="/opt/nvidia/nsight-systems/2020.4.1/bin:$PATH"
#################################################

ADD docker-configs/default /etc/nginx/sites-available/default

ADD labs/ /home/openacc/labs/
WORKDIR /home/openacc/labs
ADD scripts/entry_point.sh /home/openacc/entrypoint.sh

RUN chmod +x /home/openacc/entrypoint.sh
RUN touch /run/nginx.pid && chown -R openacc /home/openacc/ /var/lib/nginx /var/log/nginx /run/nginx.pid && (echo 'xfce4-session' > /home/openacc/.vnc/xstartup.turbovnc) && chmod 0755 /home/openacc/.vnc/xstartup.turbovnc

USER openacc

ENTRYPOINT ["/home/openacc/entrypoint.sh"]
