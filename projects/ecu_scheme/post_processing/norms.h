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

template <typename MF, typename SQ_F>
std::pair<double, lf::mesh::utils::CodimMeshDataSet<double>> H1seminorm(
    const std::shared_ptr<const lf::mesh::Mesh>& mesh_p, const MF& f,
    const SQ_F& sq_f, const lf::quad::QuadRule& quad_rule) {
  // Store intermediate squared sums of cells
  double glob_max = 0.0;

  // Dataset for storing the cellwise integrals
  lf::mesh::utils::CodimMeshDataSet<double> cell_errors(mesh_p, 0);

  // Get reference coordinates of the cells
  const Eigen::MatrixXd local_ref_coords = quad_rule.Points();
  // const Eigen::VectorXd local_weights = quad_rule.Weights();
  const lf::base::size_type num_local_qpts = quad_rule.NumPoints();

  // Iterate over all cells
  for (const lf::mesh::Entity* e : mesh_p->Entities(0)) {
    auto values = f(*e, local_ref_coords);

    double loc_max = 0.0;
    for (lf::base::size_type i = 0; i < num_local_qpts; ++i) {
      double temp = sq_f(values.col(i));
      if (temp > loc_max) {
        loc_max = temp;
      }
      if (temp > glob_max) {
        glob_max = temp;
      }
    }
    cell_errors(*e) = std::sqrt(loc_max);
  }
  return std::make_pair(std::sqrt(glob_max), cell_errors);
}

}  // namespace ecu_scheme::post_processing

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_POST_PROCESSING_NORMS_H_
