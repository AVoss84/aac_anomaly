#!/bin/bash

file="config.toml";
#my_dir=$PWD
#echo $my_dir
cp $file "../../.streamlit/$file";
# chmod 777 $file
echo "$file has been updated!";
# Run app:
streamlit run app.py;
