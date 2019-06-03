#!/bin/bash

for m in *.pth
do 
    ./train.py --predict_oof --weights $m
done

