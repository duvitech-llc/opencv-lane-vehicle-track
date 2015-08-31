#!/bin/sh

all: main.cpp
	g++ `pkg-config --cflags opencv`-o lane main.cpp `pkg-config --libs opencv`
	mv -f ./lane ./bin/

test: all
	cd ./bin && ./lane
