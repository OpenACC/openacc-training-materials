#!/bin/bash
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LABSDIR=$BASEDIR/../labs
cd $LABSDIR
RETVAL=0
for f in $(find . -name Makefile | grep -v 'module7\|module8\|module9' | grep "solutions") ; do 
  DIR=$(dirname $f)
  echo "Testing $DIR"
  cd $DIR
  make > /dev/null
  if [[ "$?" != "0" ]] ; then
    >&2 echo "Test failed in $DIR"
    RETVAL=1
  fi
  make clean
  cd -
done
if [[ "$RETVAL" == "0" ]] ; then >&2 echo "Tests pass"; fi
exit $RETVAL
