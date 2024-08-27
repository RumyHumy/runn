#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

bool UTRunnAllocFree();
bool UTRunnTrainXor();
bool UTRunnBatchesXor();
bool UTRunnMimic();

int main()
{
	printf("\nUNIT TESTING\n");

	bool flag =
		   UTRunnAllocFree()
		&& UTRunnTrainXor()
		&& UTRunnBatchesXor();
		//&& UTRunnMimic();

	return !flag;
}
