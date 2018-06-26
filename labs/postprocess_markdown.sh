#!/bin/bash
# This script will fix up markdown readmes that were exported from 
# a jupyter notebook. Some manual editing may still be required.

if [ "$#" -ne "1" ] ; then
  echo "Usage: $0 MarkdownFile.md"
  return -1
fi
FILE="$(echo $1 | tr -d [:space])"

# Strip Post-Lab section
sed -i '/^## Post-Lab/,$d' $FILE
# Convert all python syntax highlighting to shell
sed -i 's/```python/```sh/g' $FILE
# Fix image links
sed -i 's/files\///g' $FILE
# Fix notebook links
sed -i 's/ipynb/md/g' $FILE
#TODO Fix file links
echo '!!!Fix links to show source files!!!'