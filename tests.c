#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

bool UTRunnAllocFree();
bool UTRunnXor();
bool UTRunnTrainFuncXor();

int main()
{
	printf("\nUNIT TESTING\n");

	bool flag =
		   UTRunnAllocFree()
		&& UTRunnXor()
		&& UTRunnTrainFuncXor();

	return !flag;
}
