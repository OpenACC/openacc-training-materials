#!/bin/sh
export PGI_JAVA=/usr/bin/java
#service nginx start 
(/opt/websockify/run 5901 --web=/opt/noVNC --wrap-mode=ignore -- vncserver :1 -3dwm &) && \
(/usr/sbin/nginx -c /etc/nginx/nginx.conf &) && \
(pgprof &) && \
(jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/home/openacc/labs)
