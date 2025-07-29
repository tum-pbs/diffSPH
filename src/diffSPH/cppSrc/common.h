#pragma once
#define DEV_VERSION
// #define __USE_ISOC11 1
// #include <time.h>
#ifdef __INTELLISENSE__
#define OMP_VERSION
#endif
#ifdef __INTELLISENSE__
#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME diffSPH
#endif
#endif
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <optional>
#include <tuple>
#include <utility>
// #define _OPENMP
#include <algorithm>
#ifdef OMP_VERSION
#include <omp.h>
// #include <ATen/ParallelOpenMP.h>
#endif
#ifdef TBB_VERSION
#include <ATen/ParallelNativeTBB.h>
#endif
#include <ATen/Parallel.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <ATen/core/TensorAccessor.h>


#if defined(__CUDACC__) || defined(__HIPCC__)
#define hostDeviceInline __device__ __host__ inline
#define deviceInline __device__ inline
#else
#define hostDeviceInline inline
#define deviceInline inline
#endif

// Define the traits for the pointer types based on the CUDA availability
#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename T>
using traits = torch::RestrictPtrTraits<T>;
#else
template<typename T>
using traits = torch::DefaultPtrTraits<T>;
#endif

// Define tensor accessor aliases for different cases, primiarly use ptr_t when possible
template<typename T, std::size_t dim>
using ptr_t = torch::PackedTensorAccessor32<T, dim, traits>;
template<typename T, std::size_t dim>
using cptr_t = torch::PackedTensorAccessor32<T, dim, traits>;
template<typename T, std::size_t dim>
using tensor_t = torch::TensorAccessor<T, dim, traits, int32_t>;
template<typename T, std::size_t dim>
using ctensor_t = torch::TensorAccessor<T, dim, traits, int32_t>;
template<typename T, std::size_t dim>
using general_t = torch::TensorAccessor<T, dim>;


// Simple enum to specify the support mode
enum struct supportMode{
    symmetric, gather, scatter, superSymmetric
};

// Simple helper math functions
/**
 * Calculates an integer power of a given base and exponent.
 * 
 * @param base The base.
 * @param exponent The exponent.
 * @return The calculated power.
*/
deviceInline constexpr int32_t power(const int32_t base, const int32_t exponent) {
    int32_t result = 1;
    for (int32_t i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}
/**
 * Calculates the modulo of a given number n with respect to a given modulus m.
 * Works using python modulo semantics NOT C++ modulo semantics.
 * 
 * @param n The number.
 * @param m The modulus.
 * @return The calculated modulo.
 */
deviceInline constexpr auto pymod(const int32_t n, const int32_t m) {
    return n >= 0 ? n % m : ((n % m) + m) % m;
}
/**
 * Calculates the modulo of a given number n with respect to a given modulus m.
 * Works using python modulo semantics NOT C++ modulo semantics.
 * 
 * @param n The number.
 * @param m The modulus.
 * @return The calculated modulo.
 */
template<typename scalar_t>
deviceInline auto moduloOp(const scalar_t p, const scalar_t q, const scalar_t h){
    return ((p - q + h / 2.0) - std::floor((p - q + h / 2.0) / h) * h) - h / 2.0;
}

/**
 * Calculates the distance between two points in a periodic domain.
 * 
 * @param x_i The first point.
 * @param x_j The second point.
 * @param minDomain The minimum domain bounds.
 * @param maxDomain The maximum domain bounds.
 * @param periodicity The periodicity flags.
 * @return The calculated distance.
 */
template<std::size_t dim, typename scalar_t>
deviceInline auto modDistance(ctensor_t<scalar_t,1> x_i, ctensor_t<scalar_t,1> x_j, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<bool,1> periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
template<std::size_t dim, typename scalar_t>
deviceInline auto modDistance2(ctensor_t<scalar_t,1> x_i, ctensor_t<scalar_t,1> x_j, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<bool,1> periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return sum;
}

// #define DEV_VERSION

// #ifndef __CUDACC__
/**
 * @brief Returns a packed accessor for a given tensor.
 * 
 * This function builds a C++ accessor for a given tensor, based on the specified scalar type and dimension.
 * 
 * @tparam scalar_t The scalar type of the tensor.
 * @tparam dim The dimension of the tensor.
 * @param t The input tensor.
 * @param name The name of the accessor.
 * @param cuda Flag indicating whether the tensor should be on CUDA.
 * @param verbose Flag indicating whether to print32_t verbose information.
 * @param optional Flag indicating whether the tensor is optional.
 * @return The packed accessor for the tensor.
 * @throws std::runtime_error If the tensor is not defined (and not optional), not contiguous, not on CUDA (if cuda=true), or has an incorrect dimension.
 */
template <typename scalar_t, std::size_t dim>
auto getAccessor(const torch::Tensor &t, const std::string &name, bool cuda = false, bool verbose = false, bool optional = false) {
    if (verbose) {
        std::cout << "Building C++ accessor: " << name << " for " << typeid(scalar_t).name() << " x " << dim << std::endl;
    }
    if (!optional && !t.defined()) {
        throw std::runtime_error(name + " is not defined");
    }
    if (optional && !t.defined()) {
        return t.template packed_accessor32<scalar_t, dim, traits>();
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(name + " is not contiguous");
    }
    if (cuda && (t.device().type() != c10::kCUDA)) {
        throw std::runtime_error(name + " is not on CUDA");
    }

    if (t.dim() != dim) {
        throw std::runtime_error(name + " is not of the correct dimension " + std::to_string(t.dim()) + " vs " + std::to_string(dim));
    }
    return t.template packed_accessor32<scalar_t, dim, traits>();
}
// #endif

template<class F,class Tuple, std::size_t... I>
decltype(auto)
    apply_impl(F&& f, int32_t i, Tuple&& t, std::index_sequence<I...>)
{
    return f(i, std::get<I>(std::forward<Tuple>(t))...);
}
template<class F, class Tuple>
decltype(auto) invoke(F&& f, int32_t i, Tuple&& t)
{
    return apply_impl(
        std::forward<F>(f), i, std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}

template<class F,class Tuple, std::size_t... I>
decltype(auto)
    apply_implb(F&& f, bool i, Tuple&& t, std::index_sequence<I...>)
{
    return f(i, std::get<I>(std::forward<Tuple>(t))...);
}
template<class F, class Tuple>
decltype(auto) invoke_bool(F&& f, bool i, Tuple&& t)
{
    return apply_implb(
        std::forward<F>(f), i, std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}



template<typename Func, typename... Ts>
auto parallelCall(
    Func&& f,
    int32_t from, int32_t to,
    Ts&&... args
){
    // for(int32_t i = from; i < to; ++i){
    //     invoke(f, i, std::forward<Ts>(args)...);
    // }
    // return;
    #ifdef OMP_VERSION
    #pragma omp parallel for
    for(int32_t i = from; i < to; ++i){
    #else
    at::parallel_for(from, to, 0, [&](int32_t start, int32_t end){
        for(int32_t i = start; i < end; ++i){
    #endif
        invoke(f, i, std::forward<Ts>(args)...);
    }
    #ifndef OMP_VERSION
    });
    #endif
}

#ifndef DEV_VERSION
#define DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, x, ...) \
{\
    if(dim == 1){\
        constexpr int32_t dim_v = 1;\
        AT_DISPATCH_FLOATING_TYPES(scalar, x, __VA_ARGS__); \
    }else if(dim == 2){\
        constexpr int32_t dim_v = 2;\
        AT_DISPATCH_FLOATING_TYPES(scalar, x, __VA_ARGS__);\
    }else if(dim == 3){\
        constexpr int32_t dim_v = 3;\
        AT_DISPATCH_FLOATING_TYPES(scalar, x, __VA_ARGS__);\
    }else{\
        throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));\
    }\
}
#else
#define DISPATCH_FUNCTION_DIM_SCALAR( dim, scalar, x, ...) \
{\
    if (dim != 2) \
        throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim)); \
    constexpr int32_t dim_v = 2; \
    using scalar_t = float; \
    __VA_ARGS__();\
}
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>

inline void cuda_error_check() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

template<typename Func, typename... Ts>
__global__ void kernelWrapper(Func f, int32_t threads, Ts... args) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < threads){
        f(i, args...);
    }
}

template<class F,class Tuple, std::size_t... I>
decltype(auto)
launchKernel_impl(F&& f, int32_t numThreads, Tuple&& t, std::index_sequence<I...>)
{
    int32_t blockSize;  // Number of threads per block
    int32_t minGridSize;  // Minimum number of blocks required for the kernel
    int32_t gridSize;  // Number of blocks to use

    // Compute the maximum potential block size for the kernel
    auto kernel = kernelWrapper<F, decltype(std::get<I>(std::forward<Tuple>(t)))...>;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
    // cuda_error_check();
    gridSize = (numThreads + blockSize - 1) / blockSize;

    return kernelWrapper<<<gridSize, blockSize>>>(f, numThreads, std::get<I>(std::forward<Tuple>(t))...);
}

template<typename Func, class Tuple>
void launchKernel(Func kernel, int32_t numParticles, Tuple&& t) {
    launchKernel_impl(kernel, numParticles, std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}
#endif



template <typename scalar_t, std::size_t dim>
auto getAccessor(const std::optional<torch::Tensor> &t, const std::string &name, bool cuda = false, bool verbose = false, bool optional = false) {
    if (verbose) {
        std::cout << "Building C++ accessor: " << name << " for " << typeid(scalar_t).name() << " x " << dim << std::endl;
    }
    if (!optional && !t.has_value()) {
        throw std::runtime_error(name + " is not defined");
    }
    if (optional && !t.has_value()) {
        return t.value().template packed_accessor32<scalar_t, dim, traits>();
    }
    if (t.has_value() && !t.value().defined()) {
        throw std::runtime_error(name + " is not defined");
    }
    if (optional && t.has_value() && !t.value().is_contiguous()) {
        throw std::runtime_error(name + " is not contiguous");
    }
    if (optional && t.has_value() && cuda && t.value().device().type() != c10::kCUDA) {
        throw std::runtime_error(name + " is not on CUDA");
    }
    if (optional && t.has_value() && t.value().dim() != dim) {
        throw std::runtime_error(name + " is not of the correct dimension " + std::to_string(t.value().dim()) + " vs " + std::to_string(dim));
    }
    if (t.has_value()) {
        return t.value().template packed_accessor32<scalar_t, dim, traits>();
    }
}