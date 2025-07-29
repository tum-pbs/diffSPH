#include <algorithm>
#include <atomic>
#include <optional>

// AUTO GENERATED CODE BELOW
#include "gradientVector.h"

template <typename... Ts>
auto gradientVector_cpu(int32_t nQuery, int32_t dim, c10::ScalarType scalar, bool isCuda, Ts&&... args) {
    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "gradientVector", [&]() {
        auto functionArguments = invoke_bool(gradientVector_getFunctionArguments<scalar_t>, isCuda, args...);
        parallelCall(gradientVector_impl<dim_v, scalar_t>, 0, nQuery, functionArguments);
    });
}
torch::Tensor TORCH_EXTENSION_NAME::gradientVector(gradientVector_pyArguments_t) {
// Get the dimensions of the input tensors
	int32_t nQuery = positions_a.size(0);
	int32_t dim = positions_a.size(1);
	int32_t nSorted = positions_a.size(0);

// Create the default options for created tensors
	auto defaultOptions = at::TensorOptions().device(positions_a.device());
	auto hostOptions = at::TensorOptions();

// AUTO GENERATED CODE ABOVE, WILL GET OVERRIDEN

    // Allocate output tensor for the neighbor counters
    auto output = torch::zeros({nQuery, dim, dim}, defaultOptions.dtype(positions_a.dtype()));


// AUTO GENERATED CODE BELOW - Part 2
	auto wrappedArguments = std::make_tuple(gradientVector_arguments_t);

    if (positions_a.is_cuda()) {
#ifndef WITH_CUDA
        throw std::runtime_error("CUDA support is not available in this build");
#else
        std::apply(gradientVector_cuda, wrappedArguments);
#endif
    } else {
        gradientVector_cpu(nQuery, dim, positions_a.scalar_type(), positions_a.is_cuda(), wrappedArguments);
    }

// AUTO GENERATED CODE ABOVE, WILL GET OVERRIDEN - Part 2
    return output;
}
