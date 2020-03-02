#!/bin/bash

cd "$(dirname "$0")" # cd to directory of this script

# $1 is filename
# $2 is expected sha
check_sha1() {
    computed=$(sha1sum "$1" 2>/dev/null | awk '{print $1}') || return 1
    if [ "$computed" == "$2" ]; then
        return 0;
    else
        return 1;
    fi
}

# $1 is URL
# $2 is extracted file name
# $3 is the checksum
fetch() {
    if check_sha1 $2 $3; 
    then
        echo "already downloaded model"
        return
    fi
    echo "downloading $1"
    wget -q $1 -O $2
    if check_sha1 $2 $3; then
        echo "downloaded $2"
    else
        echo "HASH MISMATCH, SHA1($2) != $3"
    fi
}

fetch \
    "https://github.com/wielandbrendel/adaptive_attacks_paper/releases/download/v0.1/cifar10_conv_vae_fea_F_mid.pkl" \
    "models/cifar10_conv_vae_fea_F_mid.pkl" \
    fa7fcb32746c30e450b047cf6baf3ae647be7d1f 

fetch \
    "https://github.com/wielandbrendel/adaptive_attacks_paper/releases/download/v0.1/cifar10vgg.h5" \
    "models/cifar10vgg.h5" \
    3d62c7aead897aebddce82516f39646f551df89a     
