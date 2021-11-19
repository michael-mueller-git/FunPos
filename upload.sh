#!/bin/bash

ssh vagrant@192.168.121.25 'rm -rf /vagrant/data/FunPos'
scp -r "$PWD" vagrant@192.168.121.25:/vagrant/data
ssh vagrant@192.168.121.25 'rm -rf /vagrant/data/FunPos/checkpoint'
