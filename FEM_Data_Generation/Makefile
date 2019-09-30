# Location of Eigen lib
# EIGEN_PATH ?= ./eigen3
INCLUDES  := -I ./eigen3 
CC = g++
CFLAGS = -I. -std=c++11 -fopenmp

all: build

build: rdfem

BasicDataStructure.o: BasicDataStructure.cpp
	$(CC) $(INCLUDES) -c BasicDataStructure.cpp $(CFLAGS)

PDE.o: PDE.cpp
	$(CC) $(INCLUDES) -c PDE.cpp $(CFLAGS)

main.o: main.cpp
	$(CC) $(INCLUDES) -c main.cpp $(CFLAGS)

rdfem: BasicDataStructure.o PDE.o main.o
	$(CC) BasicDataStructure.o PDE.o main.o -o rdfem $(CFLAGS)

run: build

clean:
	rm *o rdfem
