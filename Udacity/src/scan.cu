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
void inclusiveScanKernel(T* d_out, T* d_inter, T* d_in, size_t size)
{
	extern __shared__ T sharedMem[];

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	sharedMem[tid] = idx<size?d_in[idx]:Operator<T>::identityElem();
	__syncthreads();

	if(idx < size)
	{
		/*
		 * For each element add the element 2**j left to it
		 * if it exists. Note, here `j` is the iteration count
		 * of the algorithm. 0<=j<=log(size)
		 */
		for(int offset=1; offset<blockDim.x; offset<<=1)
		{
			if(tid-offset>=0)
			{
				sharedMem[tid] = Operator<T>::apply(sharedMem[tid], sharedMem[tid-offset]);
			}
			__syncthreads();
		}

		d_out[idx] = sharedMem[tid];
		__syncthreads();

		if(tid == blockDim.x - 1)
			d_inter[bid] = sharedMem[tid];
	}
}

template<typename T, template<class> class Operator>
__global__
void addScannedValstoBlocks(T* d_out, T* d_inter, size_t size)
{
	int bid = blockIdx.x;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(bid == 0)
		return;

	T val = d_inter[bid-1];

	if(idx<size)
		d_out[idx] = Operator<T>::apply(d_out[idx], val);
}

template<typename T, template<class> class Operator>
void _inclusiveScan(T* d_out, T* d_in, size_t size)
{
	T* d_inter;

	/* Block size = 1024
	 * Grid size = ceil(size/1024)
	 */
	const size_t BLOCK_SIZE = 1024;
	const size_t GRID_SIZE = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	const size_t SHARED_MEMSIZE = BLOCK_SIZE*sizeof(T);

	checkCudaErrors(cudaMalloc((void**)&d_inter, GRID_SIZE*sizeof(T)),
					"cudaMalloc d_inter");

	inclusiveScanKernel<T, Operator><<<GRID_SIZE, BLOCK_SIZE, SHARED_MEMSIZE>>>(d_out, d_inter, d_in, size);
	checkCudaErrorKernel("inclusiveKernel");

	if(GRID_SIZE==1)
		return;

	_inclusiveScan<T, Operator>(d_inter, d_inter, GRID_SIZE);

	addScannedValstoBlocks<T, Operator><<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_inter, size);
	checkCudaErrorKernel("addScannedValstoBlocks");

	checkCudaErrors(cudaFree(d_inter), "cudaFree d_inter");
}

template<typename T, template<class> class Operator>
void inclusiveScan(T* h_out, T* h_in, size_t size)
{
	T* d_out;
	T* d_in;

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


	_inclusiveScan<T, Operator>(d_out, d_in, size);

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
	__host__ __device__
	static T identityElem()
	{
		return static_cast<T>(0);
	}
};

void test_inclusiveScan()
{
	size_t size = 1<<20;
	int* h_in = new int[size];
	int* h_out = new int[size];;
	int* h_actual = new int[size];;

	for(int i=0; i<size; i++)
		h_in[i] = 1;

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


