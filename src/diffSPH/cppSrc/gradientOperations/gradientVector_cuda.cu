#include "gradientVector.h"

void gradientVector_cuda(gradientVector_functionArguments_t) {
    int32_t nQuery = positions_a_.size(0);
    auto scalar = positions_a_.scalar_type();
    auto dim = positions_a_.size(1);

    auto wrappedArguments = std::make_tuple(positions_a_.is_cuda(), gradientVector_arguments_t_);

    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "gradientVector", [&]() {
        auto functionArguments = std::apply(gradientVector_getFunctionArguments<scalar_t>, wrappedArguments);
        launchKernel([] __device__(auto... args) { gradientVector_impl<dim_v, scalar_t>(args...); }, nQuery, functionArguments);
    });
}
