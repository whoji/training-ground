#!/bin/bash

echo 1 > ct.txt

# running the python script in background
python inc.py &

# monitor the ct.txt counter file
while :
do
  #ct=cat ./ct.txt
  ct=$(cat ct.txt)
  echo "$(cat ct.txt)"
  echo "lalala...$ct"
  #if [ $ct -gt $y ];
  if [ $ct  -gt '5' ];
  then
    echo "kill .. kill .. kill"
    pkill python
    break
  fi
  sleep 1
done

echo "all done!"