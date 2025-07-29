#pragma once
#include <common.h>
#include <algorithm>
#include <optional>
#include <atomic>


/// Begin the definitions for auto generating the function arguments:
/// functionName = gradientVector
/** BEGIN TOML
neighborIndices.type = "tensor[int64_t]"
neighborCounters.type = "tensor[int32_t]"
neighborOffsets.type = "tensor[int32_t]"
neighborListLength.type = "int64_t"

domainMin.type = "tensor[scalar_t]"
domainMax.type = "tensor[scalar_t]"
periodicity.type = "tensor[bool]"

positions_a = {type = "tensor[scalar_t]", dim = 2, reference = true}
positions_b = {type = "tensor[scalar_t]", dim = 2}
supports_a  = {type = "tensor[scalar_t]", dim = 1}
supports_b  = {type = "tensor[scalar_t]", dim = 1}

masses_a = {type = "tensor[scalar_t]", dim = 1}
masses_b = {type = "tensor[scalar_t]", dim = 1}

densities_a = {type = "tensor[scalar_t]", dim = 1}
densities_b = {type = "tensor[scalar_t]", dim = 1}

apparentAreas_a = {type = "tensor[scalar_t]", dim = 1, optional = true}
apparentAreas_b = {type = "tensor[scalar_t]", dim = 1, optional = true}

quantities_a = {type = "tensor[scalar_t]",  dim = 2, optional = true}
quantities_b = {type = "tensor[scalar_t]",  dim = 2, optional = true}
quantities_ab = {type = "tensor[scalar_t]", dim = 2, optional = true}

r_ij = {type = "tensor[scalar_t]", dim = 1}
x_ij = {type = "tensor[scalar_t]", dim = 2}

W_i = {type = "tensor[scalar_t]", dim = 1}
W_j = {type = "tensor[scalar_t]", dim = 1}
gradW_i = {type = "tensor[scalar_t]", dim = 2}
gradW_j = {type = "tensor[scalar_t]", dim = 2}

supportScheme.type = "int64_t"
gradientMode.type = "int64_t"
useApparentArea.type = "bool"
verbose.type = "bool"

output = {type = "tensor[scalar_t]", pythonArg = false, const = false, dim = 3, output = true}

*/ // END TOML

// DEF PYTHON BINDINGS
#define gradientVector_pyArguments_t torch::Tensor neighborIndices, torch::Tensor neighborCounters, torch::Tensor neighborOffsets, int64_t neighborListLength, torch::Tensor domainMin, torch::Tensor domainMax, torch::Tensor periodicity, torch::Tensor positions_a, torch::Tensor positions_b, torch::Tensor supports_a, torch::Tensor supports_b, torch::Tensor masses_a, torch::Tensor masses_b, torch::Tensor densities_a, torch::Tensor densities_b, std::optional<torch::Tensor> apparentAreas_a, std::optional<torch::Tensor> apparentAreas_b, std::optional<torch::Tensor> quantities_a, std::optional<torch::Tensor> quantities_b, std::optional<torch::Tensor> quantities_ab, torch::Tensor r_ij, torch::Tensor x_ij, torch::Tensor W_i, torch::Tensor W_j, torch::Tensor gradW_i, torch::Tensor gradW_j, int64_t supportScheme, int64_t gradientMode, bool useApparentArea, bool verbose
// DEF FUNCTION ARGUMENTS
#define gradientVector_functionArguments_t torch::Tensor neighborIndices_, torch::Tensor neighborCounters_, torch::Tensor neighborOffsets_, int64_t neighborListLength_, torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_, torch::Tensor positions_a_, torch::Tensor positions_b_, torch::Tensor supports_a_, torch::Tensor supports_b_, torch::Tensor masses_a_, torch::Tensor masses_b_, torch::Tensor densities_a_, torch::Tensor densities_b_, std::optional<torch::Tensor> apparentAreas_a_, std::optional<torch::Tensor> apparentAreas_b_, std::optional<torch::Tensor> quantities_a_, std::optional<torch::Tensor> quantities_b_, std::optional<torch::Tensor> quantities_ab_, torch::Tensor r_ij_, torch::Tensor x_ij_, torch::Tensor W_i_, torch::Tensor W_j_, torch::Tensor gradW_i_, torch::Tensor gradW_j_, int64_t supportScheme_, int64_t gradientMode_, bool useApparentArea_, bool verbose_, torch::Tensor output_
// DEF COMPUTE ARGUMENTS
#define gradientVector_computeArguments_t cptr_t<int64_t, 1> neighborIndices, cptr_t<int32_t, 1> neighborCounters, cptr_t<int32_t, 1> neighborOffsets, int64_t neighborListLength, cptr_t<scalar_t, 1> domainMin, cptr_t<scalar_t, 1> domainMax, cptr_t<bool, 1> periodicity, cptr_t<scalar_t, 2> positions_a, cptr_t<scalar_t, 2> positions_b, cptr_t<scalar_t, 1> supports_a, cptr_t<scalar_t, 1> supports_b, cptr_t<scalar_t, 1> masses_a, cptr_t<scalar_t, 1> masses_b, cptr_t<scalar_t, 1> densities_a, cptr_t<scalar_t, 1> densities_b, cptr_t<scalar_t, 1> apparentAreas_a, cptr_t<scalar_t, 1> apparentAreas_b, cptr_t<scalar_t, 2> quantities_a, cptr_t<scalar_t, 2> quantities_b, cptr_t<scalar_t, 2> quantities_ab, cptr_t<scalar_t, 1> r_ij, cptr_t<scalar_t, 2> x_ij, cptr_t<scalar_t, 1> W_i, cptr_t<scalar_t, 1> W_j, cptr_t<scalar_t, 2> gradW_i, cptr_t<scalar_t, 2> gradW_j, int64_t supportScheme, int64_t gradientMode, bool useApparentArea, bool verbose, ptr_t<scalar_t, 3> output
// DEF ARGUMENTS
#define gradientVector_arguments_t  neighborIndices,  neighborCounters,  neighborOffsets,  neighborListLength,  domainMin,  domainMax,  periodicity,  positions_a,  positions_b,  supports_a,  supports_b,  masses_a,  masses_b,  densities_a,  densities_b,  apparentAreas_a,  apparentAreas_b,  quantities_a,  quantities_b,  quantities_ab,  r_ij,  x_ij,  W_i,  W_j,  gradW_i,  gradW_j,  supportScheme,  gradientMode,  useApparentArea,  verbose,  output
#define gradientVector_arguments_t_  neighborIndices_,  neighborCounters_,  neighborOffsets_,  neighborListLength_,  domainMin_,  domainMax_,  periodicity_,  positions_a_,  positions_b_,  supports_a_,  supports_b_,  masses_a_,  masses_b_,  densities_a_,  densities_b_,  apparentAreas_a_,  apparentAreas_b_,  quantities_a_,  quantities_b_,  quantities_ab_,  r_ij_,  x_ij_,  W_i_,  W_j_,  gradW_i_,  gradW_j_,  supportScheme_,  gradientMode_,  useApparentArea_,  verbose_,  output_

// END PYTHON BINDINGS
/// End the definitions for auto generating the function arguments
// GENERATE AUTO ACCESSORS
template<typename scalar_t = float>
auto gradientVector_getFunctionArguments(bool useCuda, gradientVector_functionArguments_t){
	auto neighborIndices = getAccessor<int64_t, 1>(neighborIndices_, "neighborIndices", useCuda, verbose_);
	auto neighborCounters = getAccessor<int32_t, 1>(neighborCounters_, "neighborCounters", useCuda, verbose_);
	auto neighborOffsets = getAccessor<int32_t, 1>(neighborOffsets_, "neighborOffsets", useCuda, verbose_);
	auto neighborListLength = neighborListLength_;
	auto domainMin = getAccessor<scalar_t, 1>(domainMin_, "domainMin", useCuda, verbose_);
	auto domainMax = getAccessor<scalar_t, 1>(domainMax_, "domainMax", useCuda, verbose_);
	auto periodicity = getAccessor<bool, 1>(periodicity_, "periodicity", useCuda, verbose_);
	auto positions_a = getAccessor<scalar_t, 2>(positions_a_, "positions_a", useCuda, verbose_);
	auto positions_b = getAccessor<scalar_t, 2>(positions_b_, "positions_b", useCuda, verbose_);
	auto supports_a = getAccessor<scalar_t, 1>(supports_a_, "supports_a", useCuda, verbose_);
	auto supports_b = getAccessor<scalar_t, 1>(supports_b_, "supports_b", useCuda, verbose_);
	auto masses_a = getAccessor<scalar_t, 1>(masses_a_, "masses_a", useCuda, verbose_);
	auto masses_b = getAccessor<scalar_t, 1>(masses_b_, "masses_b", useCuda, verbose_);
	auto densities_a = getAccessor<scalar_t, 1>(densities_a_, "densities_a", useCuda, verbose_);
	auto densities_b = getAccessor<scalar_t, 1>(densities_b_, "densities_b", useCuda, verbose_);
	auto r_ij = getAccessor<scalar_t, 1>(r_ij_, "r_ij", useCuda, verbose_);
	auto x_ij = getAccessor<scalar_t, 2>(x_ij_, "x_ij", useCuda, verbose_);
	auto W_i = getAccessor<scalar_t, 1>(W_i_, "W_i", useCuda, verbose_);
	auto W_j = getAccessor<scalar_t, 1>(W_j_, "W_j", useCuda, verbose_);
	auto gradW_i = getAccessor<scalar_t, 2>(gradW_i_, "gradW_i", useCuda, verbose_);
	auto gradW_j = getAccessor<scalar_t, 2>(gradW_j_, "gradW_j", useCuda, verbose_);
	auto supportScheme = supportScheme_;
	auto gradientMode = gradientMode_;
	auto useApparentArea = useApparentArea_;
	auto verbose = verbose_;
	auto output = getAccessor<scalar_t, 3>(output_, "output", useCuda, verbose_);
	auto apparentAreas_a = getAccessor<scalar_t, 1>(apparentAreas_a_.value(), "apparentAreas_a", useCuda, verbose_, true);
	auto apparentAreas_b = getAccessor<scalar_t, 1>(apparentAreas_b_.value(), "apparentAreas_b", useCuda, verbose_, true);
	auto quantities_a = getAccessor<scalar_t, 2>(quantities_a_.value(), "quantities_a", useCuda, verbose_, true);
	auto quantities_b = getAccessor<scalar_t, 2>(quantities_b_.value(), "quantities_b", useCuda, verbose_, true);
	auto quantities_ab = getAccessor<scalar_t, 2>(quantities_ab_.value(), "quantities_ab", useCuda, verbose_, true);
	return std::make_tuple(gradientVector_arguments_t);
}
// END GENERATE AUTO ACCESSORS
// AUTO GENERATE FUNCTIONS
namespace TORCH_EXTENSION_NAME {
	torch::Tensor gradientVector(gradientVector_pyArguments_t);
}
void gradientVector_cuda(gradientVector_functionArguments_t);
template<std::size_t dim = 2, typename scalar_t = float>
deviceInline auto gradientVector_impl(int32_t i, gradientVector_computeArguments_t){
// END AUTO GENERATE FUNCTIONS
// END OF CODE THAT IS PROCESSED BY AUTO-GENERATION
    auto neighborOffset = neighborOffsets[i];

    std::array<scalar_t, dim * dim> grad;
    for (int32_t d = 0; d < dim; ++d){
        for (int32_t dd = 0; dd < dim; ++dd){
            grad[d * dim + dd] = 0;
        }
    }

    auto gradW_ii = gradW_i[i];
    auto q_i = quantities_a[i];

    for (int32_t j_ = 0; j_ < neighborCounters[i]; ++j_){
        auto j = neighborIndices[neighborOffset + j_];
        auto gradW_jj = gradW_j[j];
        auto q_j = quantities_b[j];
        auto m_j = masses_b[j];
        auto rho_j = densities_b[j];

        auto factor = m_j / rho_j;
        for (int32_t d = 0; d < dim; ++d){
            for (int32_t dd = 0; dd < dim; ++dd){
                auto gradTerm = (gradW_ii[d * dim + dd] + gradW_jj[d * dim + dd]) / 2;
                auto qTerm = q_j[dd] - q_i[dd];
                grad[d * dim + dd] += factor * gradTerm * qTerm;
            }
        }
    }
    for (int32_t d = 0; d < dim; ++d){
        for (int32_t dd = 0; dd < dim; ++dd){
            output[i][d][dd] = grad[d * dim + dd];
        }
    }
}
