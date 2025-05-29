#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/python.h>

#include "cuda_api.h"
#include "static_switch.h"

// Kernel to copy existing cache segments and insert t new rows from state
// for each head, at positions specified by cu_headlens.
// state_ptr has shape (head_num * t, dim)

template<typename scalar_t, int kblock_size=1024>
__global__ void update_flatten_view_kernel(
    scalar_t* dst_ptr,
    const scalar_t* src_ptr,
    const scalar_t* state_ptr,
    const int* headlens,
    const int* cu_headlens,
    int t,
    int dim) {
    int head_idx = blockIdx.x;
    int thread_group = blockIdx.y;
    int tid = threadIdx.x + thread_group * blockDim.x;
    int num_threads = blockDim.x * gridDim.y;

    int headlen = headlens[head_idx];

    // Offsets (in elements) for src, dst, and insertion
    int src_off_rows = cu_headlens[head_idx];                // A rows before this segment
    int dst_off_rows = src_off_rows + head_idx * t;          // account for inserted rows so far
    int insert_off_rows = cu_headlens[head_idx + 1] + head_idx * t; // insertion point after A rows

    const scalar_t* old_ptr = src_ptr + src_off_rows * dim;
    scalar_t* new_ptr = dst_ptr + dst_off_rows * dim;
    scalar_t* insert_dst_ptr = dst_ptr + insert_off_rows * dim;
    const scalar_t* insert_src_ptr = state_ptr + head_idx * t * dim;

    // Copy existing A segment
    int total_elems = headlen * dim;
    for (int base = 0; base < total_elems; base += kblock_size * num_threads) {
        int idx = base + tid * kblock_size;
        scalar_t* dst_block = new_ptr + idx;
        const scalar_t* src_block = old_ptr + idx;
        #pragma unroll
        for (int i = 0; i < kblock_size; ++i) {
            if (idx + i >= total_elems) break;
            dst_block[i] = src_block[i];
        }
    }

    // Insert t rows from state
    int insert_total = t * dim;
    for (int base = 0; base < insert_total; base += kblock_size * num_threads) {
        int idx = base + tid * kblock_size;
        scalar_t* dst_block = insert_dst_ptr + idx;
        const scalar_t* src_block = insert_src_ptr + idx;
        #pragma unroll
        for (int i = 0; i < kblock_size; ++i) {
            if (idx + i >= insert_total) break;
            dst_block[i] = src_block[i];
        }
    }
}

torch::Tensor update_flatten_view(
    torch::Tensor &cache,
    torch::Tensor &state,
    torch::Tensor &headlens,
    torch::Tensor &cu_headlens) {
  TORCH_CHECK(headlens.dtype() == torch::kInt32, "expected headlens to be int32");
  TORCH_CHECK(cu_headlens.dtype() == torch::kInt32, "expected cu_headlens to be int32");

  auto cache_shape = cache.sizes();
  int origin_len = cache_shape[0];
  int dim = cache_shape[1];
  int head_num = headlens.size(0);

  // Determine t (number of rows to insert per head)
  int total_state_rows = state.size(0);
  TORCH_CHECK(total_state_rows % head_num == 0, "state rows must be divisible by head count");
  int t = total_state_rows / head_num;

  // Allocate output: original rows + head_num * t inserted rows
  torch::Tensor out = torch::empty({origin_len + head_num * t, dim}, cache.options());

  const int kblock_size = 1;            // tune for your hardware
  const int num_threads_group = 1024;    // number of thread blocks in Y
  const int num_threads = 128;           // threads per block
  TORCH_CHECK(num_threads >= dim, "num threads should >= head dim");

  dim3 grid(head_num, num_threads_group);
  dim3 block(num_threads);

  FP16_SWITCH(cache.dtype() == torch::kFloat16, [&] {
      using scalar_t = elem_type;
      auto kernel = update_flatten_view_kernel<scalar_t, kblock_size>;
      kernel<<<grid, block, 0>>>(
          out.data_ptr<scalar_t>(),
          cache.data_ptr<scalar_t>(),
          state.data_ptr<scalar_t>(),
          headlens.data_ptr<int>(),
          cu_headlens.data_ptr<int>(),
          t,
          dim);
  });

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_flatten_view", &update_flatten_view,
        "Update flattened cache view by inserting t rows per head");
}
