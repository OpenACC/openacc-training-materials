#!/bin/bash
DIRNAME=$( dirname "${BASH_SOURCE[0]}" )
if [[ "$#" == "0" ]] ; then
  echo "Usage: ${BASH_SOURCE[0]} <jupyter notebook files"
fi
FILES=$*
for file in $FILES; do
  jupyter nbconvert --config $DIRNAME/jupyter_nbconvert_config.py $file
done
