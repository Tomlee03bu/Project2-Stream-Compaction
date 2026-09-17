#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upsweepKernel(int n, int* x, int d) {
            int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

            int stride = 1 << (d + 1);

            // Note we do this as a way to check the number of operations that actually exist at this depth
            int activeThreads = n / stride;

            if (threadIndex < activeThreads) {
                int k = threadIndex * stride;

                int left = k + (1 << d) - 1;
                int right = k + stride - 1;

                x[right] += x[left];
            }
        }

        __global__ void downsweepKernel(int n, int* x, int d) {
            int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

            int stride = 1 << (d + 1);

            int activeThreads = n / stride;

            if (threadIndex < activeThreads) {
                int k = threadIndex * stride;

                int left = k + (1 << d) - 1;
                int right = k + stride - 1;

                int t = x[left]; // Save left child
                x[left] = x[right];  // Set left child to this node’s value
                x[right] += t; // Set right child to old left value + this node’s value

            }
        }
        void checkCuda(const char* message) {
            cudaError_t err = cudaGetLastError();

            if (err != cudaSuccess) {
                printf("%s: %s\n", message, cudaGetErrorString(err));
            }
        }
        void scanDevice(int n, int* dev_odata, const int* dev_idata) {
            int threads = BENCHMARK_BLOCK_SIZE;
            int depth = ilog2ceil(n);
            int paddedN = 1 << depth;
            int blocks = (paddedN + threads - 1) / threads;
            int* dev_a;
            cudaMalloc((void**)&dev_a, paddedN * sizeof(int));

            cudaMemset(dev_a, 0, paddedN * sizeof(int));

            cudaMemcpy(
                dev_a,
                dev_idata,
                n * sizeof(int),
                cudaMemcpyDeviceToDevice
            );

            // Upsweep
            for (int d = 0; d < depth; d++) {
                upsweepKernel << <blocks, threads >> > (
                    paddedN,
                    dev_a,
                    d
                );
            }

            // Needed for exclusive scan
            cudaMemset(
                dev_a + paddedN - 1,
                0,
                sizeof(int)
            );

            // Downsweep
            for (int d = depth - 1; d >= 0; d--) {
                downsweepKernel << <blocks, threads >> > (
                    paddedN,
                    dev_a,
                    d
                );
            }

            cudaMemcpy(
                dev_odata,
                dev_a,
                n * sizeof(int),
                cudaMemcpyDeviceToDevice
            );

            cudaFree(dev_a);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int threads = BENCHMARK_BLOCK_SIZE;
            int depth = ilog2ceil(n);
            int paddedN = 1 << depth;
            int blocks = (paddedN + threads - 1) / threads;

            int* dev_a;

            cudaMalloc((void**)&dev_a, paddedN * sizeof(int));

            cudaMemset(
                dev_a,
                0,
                paddedN * sizeof(int)
            );

            cudaMemcpy(
                dev_a,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

            timer().startGpuTimer();

            for (int d = 0; d < depth; d++) {
                upsweepKernel << <blocks, threads >> > (
                    paddedN,
                    dev_a,
                    d
                );
            }

            cudaMemset(
                dev_a + paddedN - 1,
                0,
                sizeof(int)
            );

            for (int d = depth - 1; d >= 0; d--) {
                downsweepKernel << <blocks, threads >> > (
                    paddedN,
                    dev_a,
                    d
                );
            }

            timer().endGpuTimer();

            cudaMemcpy(
                odata,
                dev_a,
                n * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(dev_a);
        }

        

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

            // STEP 1: setup buffers 
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));

            cudaMemcpy(
                dev_idata,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

            int threads = BENCHMARK_BLOCK_SIZE;
            int blocks = (n + threads - 1) / threads;

            timer().startGpuTimer();

            Common::kernMapToBoolean << <blocks, threads >> > (
                n,
                dev_bools,
                dev_idata
            );

            // STEP 2: Run exclusive scan on temporary array 
            scanDevice(n, dev_indices, dev_bools);
            // STEP 3: Scatter
            Common::kernScatter << <blocks, threads >> > (
                n,
                dev_odata,
                dev_idata,
                dev_bools,
                dev_indices
            );

            timer().endGpuTimer();

            // STEP 4: Copy results back
            cudaMemcpy(
                odata,
                dev_odata,
                n * sizeof(int),
                cudaMemcpyDeviceToHost
            );
            
            //timer().endGpuTimer();

            int lastIndex;
            int lastBool;

            cudaMemcpy(
                &lastIndex,
                dev_indices + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaMemcpy(
                &lastBool,
                dev_bools + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );

            int count = lastIndex + lastBool;

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return count;
        }
    }
}
