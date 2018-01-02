#include "qsort_cuda.cuh"

#include <cstdio>
#include <cstdlib>

void run_qsort(unsigned int* data, unsigned int nitems) {
  CHECK_CUDA_ERR(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

  int left = 0;
  int right = nitems - 1;
  std::fprintf(stdout, "Launching kernel on the GPU\n");
  cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
  CHECK_CUDA_ERR(cudaDeviceSynchronize());
}

void initialize_data(unsigned int* dst, unsigned int nitems) {
  srand(2047);
  for (unsigned i = 0; i < nitems; i++)
    dst[i] = rand() % nitems;
}

int main() {
  int num_items;
  std::fprintf(stdout, "Type in the number of items: ");
  std::fscanf(stdin, "%d", &num_items);

  int device_count = 0, device = -1;
  CHECK_CUDA_ERR(cudaGetDeviceCount(&device_count));
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp properties;
    CHECK_CUDA_ERR(cudaGetDeviceProperties(&properties, i));
    if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5)) {
      device = i;
      std::fprintf(stdout, "Running on GPU %d (%s)\n", i, properties.name);
      break;
    }
    std::fprintf(stdout, "GPU %d (%s) does not support CUDA Dynamic Parallelism\n", i, properties.name);
  }
  if (device == -1) {
    std::fprintf(stderr, "QSortCUDASimple requires GPU devices with compute SM 3.5 or higher.  Exiting...\n");
    exit(EXIT_FAILURE);
  }

  cudaSetDevice(device);

  unsigned int *d_data = 0;

  CHECK_CUDA_ERR(cudaMallocManaged((void **)&d_data, num_items * sizeof(unsigned int)));

  std::fprintf(stdout, "Initializing data:\n");
  initialize_data(d_data, num_items);

  for (int i = 0; i < num_items; i++)
    std::fprintf(stdout, "Data [%u]: \n", d_data[i]);

  std::fprintf(stdout, "Running quicksort on %d elements\n", num_items);
  run_qsort(d_data, num_items);

  for (int i = 0; i < num_items; i++)
    std::fprintf(stdout, "Data [%u]: \n", d_data[i]);

  exit(EXIT_SUCCESS);
}
