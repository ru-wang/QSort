#ifndef QSORT_CUDA_SIMPLE_CUH_
#define QSORT_CUDA_SIMPLE_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>

#define MAX_DEPTH      16
#define INSERTION_SORT 32

#define CHECK_CUDA_ERR(func) check_cuda_error(func, #func, __FILE__, __LINE__)

template<typename T>
void check_cuda_error(T result, const char* const func, const char* const file, const int line) {
  if (result) {
    std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",
                 file, line, static_cast<unsigned int>(result), func);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

__device__ void selection_sort(unsigned int *data, int left, int right) {
  for (int i = left; i <= right; ++i) {
    unsigned min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for (int j = i + 1; j <= right; ++j) {
      unsigned val_j = data[j];

      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if (i != min_idx) {
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right, int depth) {
  // If we're too deep or there are few elements left, we use an insertion sort...
  if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
    selection_sort(data, left, right);
    return;
  }

  unsigned int *lptr = data + left;
  unsigned int *rptr = data + right;
  unsigned int  pivot = data[(left + right) / 2];

  // Do the partitioning.
  while (lptr <= rptr) {
    // Find the next left- and right-hand values to swap.
    unsigned int lval = *lptr;
    unsigned int rval = *rptr;

    // Move the left pointer as long as the pointed element is smaller than the pivot.
    while (lval < pivot) {
      lptr++;
      lval = *lptr;
    }

    // Move the right pointer as long as the pointed element is larger than the pivot.
    while (rval > pivot) {
      rptr--;
      rval = *rptr;
    }

    // If the swap points are valid, do the swap!
    if (lptr <= rptr) {
      *lptr++ = rval;
      *rptr-- = lval;
    }
  }

  // Now the recursive part.
  int nright = rptr - data;
  int nleft = lptr - data;

  // Launch a new block to sort the left part.
  if (left < (rptr - data)) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
    cudaStreamDestroy(s);
  }

  // Launch a new block to sort the right part.
  if ((lptr - data) < right) {
    cudaStream_t s1;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
    cudaStreamDestroy(s1);
  }
}


#endif  // QSORT_CUDA_SIMPLE_CUH_
