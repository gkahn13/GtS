#!/bin/bash

cmd=$1
sandboxname=default
gcgdir=`git rev-parse --show-toplevel`

if [ "$cmd" == "build" ]; then

    dockerfile=$2
    if [ -z $dockerfile ]; then
        echo "Need to provide a dockerfile to build"
    else
        echo "Building..."
        docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -f $dockerfile -t gcg-$sandboxname .
        echo "Done building"
    fi

elif [ "$cmd" == "start" ]; then
    
    echo "Starting..."
    docker run \
        -it \
        --runtime=nvidia \
        --user $(id -u):$(id -g) \
        -v "$gcgdir":/home/gcg-user/gcg \
        -v /media:/media \
        -d \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --name gcg-$sandboxname \
        gcg-$sandboxname
    echo "Done starting"

elif [ "$cmd" == "ssh" ]; then

    echo "Sshing..."
    docker exec -it gcg-$sandboxname /bin/bash
    echo "Done sshing"

elif [ "$cmd" == "stop" ]; then

    echo "Stopping..."
    docker container stop gcg-$sandboxname
    docker rm gcg-$sandboxname
    echo "Done stopping"

elif [ "$cmd" == "clean" ]; then

    echo "Cleaning..."
    docker rmi -f $(docker images -q --filter "dangling=true")
    echo "Done cleaning"

elif [ "$cmd" == "--help" ]; then

    echo "Valid commands: build, start, ssh, stop, clean"

else
    echo "INVALID"
fi
