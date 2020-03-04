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
    if [ -d models ];
    then
        echo "already downloaded models"
        return
    fi
    echo "downloading $1"
    wget -q $1 -O $2
    if check_sha1 $2 $3; then
        echo "downloaded $2"
        tar xzvf $2
    else
        echo "HASH MISMATCH, SHA1($2) != $3"
    fi
    rm -f $2
}

if [ ! -d defense ]
then
    git clone https://github.com/xuwangyin/AAT-CIFAR10.git defense 
    cd defense
    git checkout 6cbbc81
    cd ..
else
    echo "defense already downloaded"
fi

fetch \
    "https://asymmetrical-adversarial-training.s3.amazonaws.com/cifar10/checkpoints.tar.gz" \
    "checkpoints.tar.gz" \
    449953c91cda5352740e30d05dcaa6c57389d680     
