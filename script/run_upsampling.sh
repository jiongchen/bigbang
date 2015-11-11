#!/bin/bash

exe=../build/bin/test_upsampling

echo -e "\033[31m"
$exe -p coarse_offline_sim

echo -e "\033[32m"
$exe -p fine_offline_sim

echo -e "\033[33m"
$exe -p upsampling

echo -e "\033[36m"
$exe -p online_sim

echo -e "\033[0m"