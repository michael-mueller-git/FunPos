#!/bin/bash

rm -f checkpoint/FunPos_*
scp -r vagrant@192.168.121.25:/vagrant/data/FunPos/checkpoint/* checkpoint
