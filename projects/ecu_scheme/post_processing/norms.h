// Project header file include
// C system headers
// C++ standard library headers
// Other libraries headers

#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_POST_PROCESSING_NORMS_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_POST_PROCESSING_NORMS_H_

#include <lf/mesh/entity.h>
#include <lf/mesh/mesh.h>
#include <lf/mesh/utils/utils.h>
#include <lf/quad/quad.h>

#include <functional>

namespace ecu_scheme::post_processing {

/**
 * @brief Compute L2-error norm for given Mesh Function and cell-wise errors for
 * plotting purposes This implementation is used instead of the usual LehrFEM++
 * methods for the L2-error computation of Mesh Functions corresponding to
 * 1-forms
 * @tparam MF underlying type of Mesh Function
 * @tparam SQ_F type of the square function (can be for scalar or vector-valued
 * functions)
 * @param mesh_p pointer to the mesh
 * @param f Mesh Function representing the difference between the exact solution
 * and FE solution
 * @param sq_f squaring function (either applied to scalar or vector-valued
 * functions)
 * @param quad_rule qudrature rule for computing the norm
 * @return Pair containing the L2-error norm of the Mesh Function and data
 * structure of cell-wise errors for the mesh
 */
template <class MF, typename SQ_F>
std::pair<double, lf::mesh::utils::CodimMeshDataSet<double>> L2norm(
    const std::shared_ptr<const lf::mesh::Mesh>& mesh_p, const MF& f,
    const SQ_F& sq_f, const lf::quad::QuadRule& quad_rule) {
  // Store intermediate squared sums of cells
  double squared_sum = 0.0;

  // Dataset for storing the cellwise integrals
  lf::mesh::utils::CodimMeshDataSet<double> cell_errors(mesh_p, 0);

  // Get reference coordinates of the cells
  const Eigen::MatrixXd local_ref_coords = quad_rule.Points();
  const Eigen::VectorXd local_weights = quad_rule.Weights();
  const lf::base::size_type num_local_qpts = quad_rule.NumPoints();

  // Iterate over all cells
  for (const lf::mesh::Entity* e : mesh_p->Entities(0)) {
    // Get determinant of the pullback
    Eigen::VectorXd det{e->Geometry()->IntegrationElement(local_ref_coords)};
    // Get function values
    auto values = f(*e, local_ref_coords);
    // Compute local quadrature of the squared norm
    double local_squared_sum = 0.0;
    for (lf::base::size_type i = 0; i < num_local_qpts; ++i) {
      local_squared_sum += det(i) * local_weights(i) * sq_f(values[i]);
    }
    cell_errors(*e) = std::sqrt(local_squared_sum);
    squared_sum += local_squared_sum;
  }
  return std::make_pair(std::sqrt(squared_sum), cell_errors);
}

}  // namespace ecu_scheme::post_processing

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_POST_PROCESSING_NORMS_H_
