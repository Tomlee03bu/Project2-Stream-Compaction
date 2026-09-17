#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */

        void scanImpl(int n, int* odata, const int* idata) {
            //As it is exclusive scan, we do not take into consideration the first element, therefore the first element
            // of the output list should just be identity (ie 0 in the integer case)
            odata[0] = 0;

            for (int i = 1; i < n; i++) {
                // for example in i = 1, we would take the sum of the previous index in the output and add the new element
                //from the input, so it would simply be 0 + the previous element
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            //note we create this new function as when we do compact with scan we cannot call the same timer twice. 
            scanImpl(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            int count = 0;

            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // We devise two new lists, one that contains a 0 or 1 if that element is a 0 or not,
            // and a list that contains the exclusive scan of the bools list
            int* bools = new int[n];
            int* indices = new int[n];

            // Constructing our bools list
            for (int i = 0; i < n; i++) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Exclusive scanning the boolean array
            //scan(n, indices, bools);
            scanImpl(n, indices, bools);
            // We now scatter surviving elements into compacted positions
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    odata[indices[i]] = idata[i];
                }
            }

            // Count provides the number of elements remaining
            // Because this is an exclusive scan, the final scan value does not
            // include bools[n - 1]
            int count = indices[n - 1] + bools[n - 1];

            timer().endCpuTimer();
            return count;
        }
    }
}
