#include "grouped_gemm.h"
#include "permute.h"
#include "sinkhorn.h"

#include <torch/extension.h>

namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
  m.def("sinkhorn", &sinkhorn, "Sinkhorn kernel");
  m.def("permute", &moe_permute_topK_op, "Token permutation kernel");
  m.def("permute_pad", &moe_permute_topK_op_pad, "Token permutation kernel with padding support");
  m.def("unpermute", &moe_recover_topK_op, "Token un-permutation kernel");
  m.def("unpermute_unpad", &moe_recover_topK_op_unpad, "Token un-permutation kernel unpad");
  m.def("unpermute_inplace", &moe_recover_topK_op_inplace, "Token un-permutation kernel with output being inplaced");
  m.def("unpermute_bwd", &moe_recover_topK_bwd_op, "Token un-permutation backward kernel");
  m.def("unpermute_unpad_bwd", &moe_recover_topK_unpad_bwd_op, "Token un-permutation unpad backward kernel");
}

}  // namespace grouped_gemm
