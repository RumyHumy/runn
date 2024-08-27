# gcc ./runn/src/runn.c ./unit_tests/xor.c -o ./unit_tests/xor -lm -I./runn/include

CC = gcc

CFLAGS = -Wall -std=c99 -I./runn/include

LDFLAGS = -lm

SOURCES = ./runn/src/runn.c $(wildcard ./utests/*.c)

tests: ./tests.c
	$(CC) $(CFLAGS) $(LDFLAGS) ./tests.c $(SOURCES) -o ./bin/tests
	./bin/tests

clean:
	rm -f ./bin/tests
