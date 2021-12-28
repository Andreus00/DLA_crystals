CC := gcc
LIBS := -lpthread
CFLAGS := -Wall
PATH := ./src/generator/*
FILES := *.c
BIN := ./bin
OUTS := "utils single_core_dla"

.PHONY : bin
bin:
	mkdir -p $(BIN)

all: $(FILES)
	$(CC) $(CFLAGS) src/generator/serial/single_core_dla.c -o $(BIN)/a.o

*.c : ./src/generator/serial/single_core_dla.c

.PHONY : clean
clean : 
	rm -rfi $(BIN)

