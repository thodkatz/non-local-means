CC = g++
CFLAGS = -g -pedantic -Wall

BIN = bin
DIRSRC = src
DIR_v0 = $(DIRSRC)/v0
DIR_v1 = $(DIRSRC)/v1
DIR_v2 = $(DIRSRC)/v2
BASE = $(DIRSRC)/*.c 
SRC_v0 = $(DIR_v0)/*.c 
SRC_v1 = $(DIR_v1)/*.cu
SRC_v2 = $(DIR_v2)/*.cu
INC = -I include

$(shell mkdir -p bin)

v0: $(SRC_v0)
	$(CC) $(CFLAGS) $^ $(INC) -o $(BIN)/$@

v1: $(SRC_v1)
	nvcc $^ $(INC) -o $(BIN)/$@

v2: $(SRC_v2)
	nvcc $^ $(INC) -o $(BIN)/$@

.PHONY: clean v0 v1 v2

clean:
	rm -f bin/*
