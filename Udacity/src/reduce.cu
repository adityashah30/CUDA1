/*
 * The reduce algorithm.
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
template <typename T, template<typename> class Operator>
__global__
void reduceKernel(T* d_out, T* d_in, size_t size)
{

	extern __shared__ T sharedMem[];

	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int tid = threadIdx.x;

	/*
	 * If idx < size then copy the data from global memory to
	 * shared memory.
	 * Else pad the shared memory with identity element.
	 */
	sharedMem[tid] = idx<size?d_in[idx]:Operator<T>::identityElem();
	__syncthreads();

	if(idx < size)
	{
		//Now reduce the shared memory.
		/*
		 * Start with window with half the blockDim
		 * and then reduce the window by half with
		 * each iteration.
		 * Assumes that blockDim would be a power of 2.
		 */
		for(size_t s = blockDim.x/2; s>0; s>>=1)
		{
			if(tid < s)
			{
				sharedMem[tid] = Operator<T>::apply(sharedMem[tid], sharedMem[tid+s]);
			}
			__syncthreads();
		}

		//Now copy the data back to global memory.
		/*If the thread is first in the block, write the data
		 * back to the global memory. Each block would have one
		 * reduced value at the end of this kernel.
		 */
		if(tid==0)
		{
			d_out[blockIdx.x] = sharedMem[0];
		}
	}
}

template <typename T, template<typename> class Operator>
void reduce(T* h_out, T* h_in, size_t size)
{
	T* d_out;
	T* d_in;

	size_t BLOCK_SIZE = 1024;
	size_t SHAREDMEM_SIZE = BLOCK_SIZE*sizeof(T);
	size_t GRID_SIZE = (size+BLOCK_SIZE-1)/BLOCK_SIZE;

	/*
	 * Device memory allocation.
	 */
	checkCudaErrors(cudaMalloc((void**)&d_in, size*sizeof(T)),
					"cudamalloc d_in");
	checkCudaErrors(cudaMalloc((void**)&d_out, GRID_SIZE*sizeof(T)),
					"cudamalloc d_out");
	/*
	 * Host to Device memcpy
	 */
	checkCudaErrors(cudaMemcpy(d_in, h_in, size*sizeof(T), cudaMemcpyHostToDevice),
					"cudamemcpy h_in to d_in");

	/*
	 * Kernel should be launched until there is only 1
	 * grid remaining as the in the last invocation, one threadblock
	 * will reduce all the elements to produce the final result.
	 */
	do
	{
		/*
		 * Grid size = ceil(size/1024)
		 */
		GRID_SIZE = (size+BLOCK_SIZE-1)/BLOCK_SIZE;

		/*
		 * Launch the kernel instance to generate intermediate results.
		 */
		reduceKernel<T, Operator><<<GRID_SIZE, BLOCK_SIZE, SHAREDMEM_SIZE>>>(d_out, d_in, size);
		checkCudaErrorKernel("reduceKernel");

		/*
		 * For the next round of computation, there are
		 * only GRID_SIZE objects remaining, hence downscale
		 * the size.
		 */
		size = GRID_SIZE;

		/*
		 * Copy d_out to d_in for next round of computation.
		 */
		checkCudaErrors(cudaMemcpy(d_in, d_out, size*sizeof(T), cudaMemcpyDeviceToDevice),
								"cudamemcpy d_out to d_in");
	}while(GRID_SIZE!=1);

	/*
	 * Device to Host memcpy
	 */
	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost),
					"cudamemcpy d_out to h_out");

	/*
	 * Device memory deallocation
	 */
	checkCudaErrors(cudaFree(d_in), "cudafree d_in");
	checkCudaErrors(cudaFree(d_out), "cudafree d_out");
}

template <typename T>
class AddOperatorReduce
{
public:
	__host__ __device__
	static T apply(T a, T b)
	{
		return a+b;
	}
	__host__ __device__
	static T identityElem()
	{
		return static_cast<T>(0);
	}
};

void test_reduce()
{
	size_t size = 1<<20;
	float* h_in = new float[size];
	float h_out;
	float h_actual = 0;

	for(int i=0; i<size; i++)
	{
		h_in[i] = 1.0f;
		h_actual += h_in[i];
	}

	reduce<float, AddOperatorReduce>(&h_out, h_in, size);

	delete [] h_in;

	assert(h_out == h_actual);

	std::cout << "Reduce Test passed!" << std::endl;
}
