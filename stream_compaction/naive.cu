#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scanKernel(int n, const int* Adata, int* Bdata, int d) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k < n) {
                //same as the 2^(d-1), except bitwise shift. Effectively giving us 1, 2, 4, 8, etc.
                int offset = 1 << (d - 1);
                if (k >= offset) {
                    Bdata[k] = Adata[k - offset] + Adata[k];
                }
                else {
                    //since we swap between buffers, we need to actually write the values for the other indices
                    Bdata[k] = Adata[k];
                }
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //creating our two buffers
            int* dev_a;
            int* dev_b;

            //allocate
            cudaMalloc((void**)&dev_a, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            
            //assign
            cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int threads = BENCHMARK_BLOCK_SIZE;
            int blocks = (n + threads - 1) / threads;
            timer().startGpuTimer();
            for (int d = 1; d <= ilog2ceil(n); d++) {
                //scanning for that iteration
                scanKernel << <blocks, threads >> > (n, dev_a, dev_b, d);
                //swapping buffers
                int* temp = dev_a;
                dev_a = dev_b;
                dev_b = temp;
            }

            timer().endGpuTimer();

            //shifting elements
            odata[0] = 0;

            cudaMemcpy(odata + 1, dev_a, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            //free buffers
            cudaFree(dev_a);
            cudaFree(dev_b);
        }
    }
}
