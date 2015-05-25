/*
 * To implement the following algorithms on the GPU
 * 1. Map
 * 2. Transpose
 * 3. Gather
 * 4. Scatter
 * 5. Stencil
 * 6. Reduce
 * 7. Scan
 */

/*
 * Progress
 *
 * 12:04; 21 May 2015: test_reduce successful.
 */
#include <iostream>

using namespace std;

extern void test_reduce();
extern void test_inclusiveScan();

int main()
{
	test_reduce();
	test_inclusiveScan();
	return 0;
}
