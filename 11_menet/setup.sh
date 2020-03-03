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
    "https://github.com/wielandbrendel/adaptive_attacks_paper/releases/download/v0.1/ckpt.t7_ResNet18_advtrain_concat_usvt_0.5_white" \
    "models/ckpt.t7_ResNet18_advtrain_concat_usvt_0.5_white" \
    bfc94db73c1019f6eff523c4f5c2623250997cc6    

fetch \
    "https://github.com/wielandbrendel/adaptive_attacks_paper/releases/download/v0.1/ckpt.t7_ResNet18_pure_concat_usvt_0.5_white" \
    "models/ckpt.t7_ResNet18_pure_concat_usvt_0.5_white" \
    6c12aa94b74d1c3fabc90ab042837081f87a5a7e    
