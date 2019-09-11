#!/bin/bash

x=500
y=10
if [ $x -gt $y ];
then
  echo "sdfadfssaf"
fi


while :
do
  #ct=cat ./ct.txt
  ct=$(cat ct.txt)
  echo "$(cat ct.txt)"
  echo "lalala...$ct"
  #if [ $ct  -gt $y ];
  if [ $ct  -gt '5' ];
  then
    echo "kill .. kill .. kill"
    pkill python
    break
  fi
  sleep 1
done