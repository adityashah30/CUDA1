/*
 * The scan algorithm.
 * Create a generic algorithm that can be used without any
 * further modifications.
 */
#include <cstdlib>
#include <cassert>

//The header containing helper functions and classes
#include "cudafoo.h"

/*
 * Assumptions:
 * 1. The operator class is an operation that is commutative in nature.
 * 2. Operator class would have a static method named "apply" which has
 *    the following signature.
 *    	T Operator::apply(T a, T b);
 *    This returns a `op` b where `op` is the operation encapsulated by
 *    the operator.
 */
template<typename T, template<class> class Operator>
__global__
void inclusiveScanKernel(T* d_out, T* d_in, size_t size)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(idx < size)
	{
		//Copy the data from global memory to shared memory.
		d_out[idx] = d_in[idx];
		__syncthreads();

		/*
		 * For each element add the element 2**j left to it
		 * if it exists. Note, here `j` is the iteration count
		 * of the algorithm. 0<=j<=log(size)
		 */
		for(int offset=1; offset<size; offset<<=1)
		{
			if(idx-offset>=0)
			{
				d_out[idx] += d_out[idx-offset];
			}
			__syncthreads();
		}
	}
}

template<typename T, template<class> class Operator>
void inclusiveScan(T* h_out, T* h_in, size_t size)
{
	T* d_out;
	T* d_in;

	/* Block size = 1024
	 * Grid size = ceil(size/1024)
	 */
	const size_t BLOCK_SIZE = 1024;
	const size_t GRID_SIZE = (size+BLOCK_SIZE-1)/BLOCK_SIZE;

	/*
	 * Device memory allocation.
	 */
	checkCudaErrors(cudaMalloc((void**)&d_in, size*sizeof(T)),
					"cudamalloc d_in");
	checkCudaErrors(cudaMalloc((void**)&d_out, size*sizeof(T)),
						"cudamalloc d_out");

	/*
	 * Host to Device memcpy
	 */
	checkCudaErrors(cudaMemcpy(d_in, h_in, size*sizeof(T), cudaMemcpyHostToDevice),
					"cudamemcpy h_in to d_in");

	inclusiveScanKernel<T, Operator><<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, size);
	checkCudaErrorKernel("inclusiveScanKernel_1");

	/*
	 * Device to Host memcpy
	 */
	checkCudaErrors(cudaMemcpy(h_out, d_out, size*sizeof(T), cudaMemcpyDeviceToHost),
					"cudamemcpy d_out to h_out");

	/*
	 * Device memory deallocation
	 */
	checkCudaErrors(cudaFree(d_in), "cudafree d_in");
	checkCudaErrors(cudaFree(d_out), "cudafree d_out");
}

template <typename T>
class AddOperatorScan
{
public:
	__host__ __device__
	static T apply(T a, T b)
	{
		return a+b;
	}
};

void test_inclusiveScan()
{
	size_t size = 1048;
	int* h_in = new int[size];
	int* h_out = new int[size];;
	int* h_actual = new int[size];;

	for(int i=0; i<size; i++)
		h_in[i] = (i+1);

	h_actual[0] = h_in[0];
	for(int i=1; i<size; i++)
		h_actual[i] = h_actual[i-1]+h_in[i];

	inclusiveScan<int, AddOperatorScan>(h_out, h_in, size);

	for(int i=0; i<size; i++)
	{
		assert(h_actual[i]==h_out[i]);
	}

	std::cout << "Inclusive Scan Test passed!" << std::endl;

	delete h_in;
	delete h_out;
	delete h_actual;
}


