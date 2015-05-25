/*
 * Header containing helper functions, classes and macros.
 */
#ifndef _CUDAFOO_H_
#define _CUDAFOO_H_

#include <iostream>
#include <ctime>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define checkCudaErrors(a, b)             \
{ 										  \
 cudaError_t _rc;                         \
 if((_rc=a)!=cudaSuccess)                 \
 {                                        \
	std::cerr << "Error in "              \
              << b << std::endl;          \
    std::cerr << "Reason: "	              \
    		  << cudaGetErrorString(_rc)  \
    		  << std::endl;               \
	exit(1);                              \
 }                                        \
}

#define checkCudaErrorKernel(b)           \
 cudaDeviceSynchronize();                 \
 checkCudaErrors(cudaGetLastError(), b)

class CudaTimer
{
private:
	cudaEvent_t start, stop;
	float elapsedTime; //ElapsedTime in milliseconds
public:
	CudaTimer()
	{
		checkCudaErrors(cudaEventCreate(&start),
						"eventcreate start");
		checkCudaErrors(cudaEventCreate(&stop),
						"eventcreate stop");
		elapsedTime = 0.0f;
	}
	~CudaTimer()
	{
		checkCudaErrors(cudaEventDestroy(start),
						"eventdestroy start");
		checkCudaErrors(cudaEventDestroy(stop),
						"eventdestroy stop");
	}
	void startTimer()
	{
		checkCudaErrors(cudaEventRecord(start, 0),
						"eventrecord start");
	}
	void stopTimer()
	{
		checkCudaErrors(cudaEventRecord(stop, 0),
						"eventrecord stop");
		checkCudaErrors(cudaEventSynchronize(stop),
						"eventsynchronize stop");
		checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop),
						"eventelapsedtime");
	}
	float getElapsedTime()
	{
		return elapsedTime;
	}
};

class Timer
{
private:
	struct timeval start, stop;
	float elapsedTime; //ElapsedTime in milliseconds
public:
	Timer()
	{
		elapsedTime = 0.0f;
	}
	void startTimer()
	{
		gettimeofday(&start, NULL);
	}
	void stopTimer()
	{
		gettimeofday(&stop, NULL);
		elapsedTime = (stop.tv_sec-start.tv_sec)/1000 +
		  			  (stop.tv_usec-start.tv_usec)*1000;
	}
	float getElapsedTime()
	{
		return elapsedTime;
	}
};

#endif
