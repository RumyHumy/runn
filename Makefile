# Testing
tests: ut-xor

ut-xor:
	gcc unit_tests/xor.c runn/src/runn.c -o ./bin/Xor -lm -std=c99 -I./runn/include
	./bin/Xor
