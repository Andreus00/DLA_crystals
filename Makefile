CC := "gcc"
LIBS := ""
FLAGS := ""
FILES := "utils/utils.c  src/generator/serial/single_core_dla.c"
OUTS := "utils single_core_dla"

all: bin $(FILES)
	$(CC) $(FLAGS) $(LIBS) $^ -o $(OUTS)

.PHONY : bin
bin:
	mkdir -p bin

.PHONY : clean
clean : 
	rm -rfi bin

