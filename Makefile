all:
	gcc unit_tests.c runn.c -o ./Build -lm -std=c99 -I.
	./Build
