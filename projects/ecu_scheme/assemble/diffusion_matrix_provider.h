//
// 
//

#ifndef THESIS_ASSEMBLE_DIFFUSION_MATRIX_PROVIDER_H_
#define THESIS_ASSEMBLE_DIFFUSION_MATRIX_PROVIDER_H_

#include "lf/fe/scalar_fe_space.h"
#include <lf/mesh/utils/utils.h>

namespace assemble {

// todo consider if this is needed over the LehrFEM++ DiffusionElementMatrixProvider
template <typename SCALAR, typename DIFF_COEFF>
class DiffusionMatrixProvider {

 public:
  DiffusionMatrixProvider(std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space,DIFF_COEFF d);

 private:
    DIFF_COEFF diff_coeff_;
    std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space_;
};

template <typename SCALAR, typename DIFF_COEFF>
DiffusionMatrixProvider<SCALAR, DIFF_COEFF>::DiffusionMatrixProvider(
    std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space, DIFF_COEFF d)
    : fe_space_(std::move(fe_space)), diff_coeff_(std::move(d)) {}

} // assemble

#endif //THESIS_ASSEMBLE_DIFFUSION_MATRIX_PROVIDER_H_
