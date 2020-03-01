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
    if [ -d models/naturally_trained ]
    then
        echo "already downloaded model"
        return
    fi
    echo "downloading $1"
    wget -q $1 -O $2
    if check_sha1 $2 $3; then
        echo "downloaded $2"
        unzip $2
    else
        echo "HASH MISMATCH, SHA1($2) != $3"
    fi
    rm -f $2
}

if [ ! -d defense ]
then
    git clone https://github.com/yk/icml19_public.git defense 
    cd defense
    git checkout ace61a2
    cd ..
else
    echo "defense already downloaded"
fi

fetch \
    https://www.dropbox.com/s/cgzd5odqoojvxzk/natural.zip?dl=1 \
    natural.zip \
    2650dc3179123ade9fe0897a7baf47e67916ff2a
