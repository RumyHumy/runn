#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

bool UTRunnXor();
bool UTRunnAllocFree();

int main()
{
	printf("\nUNIT TESTING\n");

	bool flag =
		   UTRunnAllocFree()
		&& UTRunnXor();

	return !flag;
}
