#!/bin/sh

all: main.cpp
	g++ `pkg-config --cflags opencv`-o lane main.cpp `pkg-config --libs opencv`
	mv -f /home/acklinr/Public/ws/vision/opencv/lanetrack/lane/lane ./bin/

test: all
	cd ./bin && ./lane
