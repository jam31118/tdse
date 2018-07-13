
flags = -O3 -Wall

all: main

main: ./main.c ../../matrix.c
	$(CC) $(flags) -o $@ $^ -I../../

clean:
	$(RM) main

