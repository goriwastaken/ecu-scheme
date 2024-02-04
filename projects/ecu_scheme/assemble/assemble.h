#ifndef THESIS_ASSEMBLE_ASSEMBLE_H_
#define THESIS_ASSEMBLE_ASSEMBLE_H_

#include <convection_upwind_matrix_provider.h>
#include <diffusion_matrix_provider.h>
#include <load_vector_provider.h>
#include <lumped_mass_matrix_provider.h>
#include <supg_linear_matrix_provider.h>
#include <supg_matrix_provider.h>

namespace ecu_scheme{

  /**
   * @brief Collection of matrix and vector element providers
   */

  namespace assemble{} // namespace assemble
} // namespace ecu_scheme

#endif //THESIS_ASSEMBLE_ASSEMBLE_H_