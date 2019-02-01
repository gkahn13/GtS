#!/usr/bin/env bash

name=$1

if [[ -z "$name" ]]; then
    echo "Need to pass in the name of the sandbox"
else
    fullname="src/sandbox/${name}"

    if [ -d "$fullname" ]; then
        echo "Sandbox '${name}' already exists"
        echo "Delete this sandbox or choose a different name"
    else
        echo "Creating sandbox '${name}'"
        mkdir $fullname
        cp -r docker/ $fullname/docker
        sed "s/sandboxname=.*/sandboxname=$name/g" $fullname/docker/gcg-docker.sh -i
        touch $fullname/__init__.py
    fi

fi
