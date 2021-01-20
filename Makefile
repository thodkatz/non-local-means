CC = g++
CFLAGS = -g -pedantic -Wall

BIN = bin
DIRSRC = src
DIR_v0 = $(DIRSRC)/v0
DIR_v1 = $(DIRSRC)/v1
DIR_v2 = $(DIRSRC)/v2
BASE = $(DIRSRC)/*.c 
SRC_v0 = $(BASE) $(DIR_v0)/*.c 
SRC_v1 = $(BASE) $(DIR_v1)/*.c
SRC_v2 = $(BASE) $(DIR_v2)/*.c
INC = -I include

$(shell mkdir -p bin)

all: v0

v0: $(SRC_v0)
	$(CC) $(CFLAGS) $^ $(INC) -o $(BIN)/$@


.PHONY: clean v0

clean:
	rm -f bin/*
