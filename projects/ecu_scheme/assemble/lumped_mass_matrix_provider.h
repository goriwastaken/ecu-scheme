#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_LUMPED_MASS_MATRIX_PROVIDER_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_LUMPED_MASS_MATRIX_PROVIDER_H_

#include <algorithm>

#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"
#include <Eigen/Core>
namespace ecu_scheme::assemble {

template <typename FUNCTOR>
class LumpedMassMatrixProvider{
 public:
  LumpedMassMatrixProvider() = delete;
  ~LumpedMassMatrixProvider() = default;
  explicit LumpedMassMatrixProvider(FUNCTOR f) : f_(std::move(f)) {}

  Eigen::Matrix<double, 6, 6> Eval(const lf::mesh::Entity& entity);

  bool isActive(const lf::mesh::Entity& /*entity*/) { return true; }
 private:
  FUNCTOR f_;
};

template <typename FUNCTOR>
Eigen::Matrix<double, 6, 6> LumpedMassMatrixProvider<FUNCTOR>::Eval(const lf::mesh::Entity& entity) {
  LF_ASSERT_MSG(entity.RefEl() == lf::base::RefEl::kTria(), "Unsupported cell type");
  const lf::geometry::Geometry* geom = entity.Geometry();
  Eigen::Matrix<double, 6, 6> elem_mat;
  elem_mat.setZero();

  const Eigen::MatrixXd corners = lf::geometry::Corners(*geom);
  const double area = lf::geometry::Volume(*geom);

  // Evaluate element matrix based on the 2d trapezoidal rule
  for(int i = 0; i < 6; ++i) {
    elem_mat(i, i) = area / 3.0 * f_(corners.col(i));
  }

  return elem_mat;
}

}  // namespace ecu_scheme

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_LUMPED_MASS_MATRIX_PROVIDER_H_
