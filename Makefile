all:
	gcc unit_tests.c runn/src/runn.c -o ./Build -lm -std=c99 -I./runn/include
	./Build
