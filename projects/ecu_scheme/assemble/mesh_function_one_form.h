
#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_MESH_FUNCTION_ONE_FORM_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_MESH_FUNCTION_ONE_FORM_H_

#include <lf/uscalfe/uscalfe.h>

namespace ecu_scheme::assemble {

/**
 * @brief Class representing the implementation of a Mesh Function for 1-forms
 * @tparam SCALAR the SCALAR type of the coefficient vector
 */
template <typename SCALAR>
class MeshFunctionOneForm {
 public:
  /**
   * @brief Constructor for Mesh Function of 1-forms
   * @param mu basis function coefficient vector in global coordinates
   * @param mesh_p underlying mesh
   */
  MeshFunctionOneForm(const Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>& mu,
                      const std::shared_ptr<const lf::mesh::Mesh>& mesh_p)
      : mu_(mu), mesh_p_(mesh_p) {}
  /**
   * @brief Evaluates the basis function of 1-forms at local coordinates of a
   * cell This implementation concerns only first order edge elements on
   * triangular cells
   * @param entity reference to the cell
   * @param local_coords local coordinates of the cell
   * @return vector of the sum of edge element basis functions at local
   * coordinates
   */
  std::vector<Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>> operator()(
      const lf::mesh::Entity& entity,
      const Eigen::MatrixXd& local_coords) const;

 private:
  Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> mu_;
  std::shared_ptr<const lf::mesh::Mesh> mesh_p_;
};

template <typename SCALAR>
std::vector<Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>>
MeshFunctionOneForm<SCALAR>::operator()(
    const lf::mesh::Entity& entity, const Eigen::MatrixXd& local_coords) const {
  LF_ASSERT_MSG(entity.RefEl() == lf::base::RefEl::kTria(),
                "Only implemented for triangular cells");

  std::vector<Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>> result(
      local_coords.cols());

  // Get geometric information about the cell
  const lf::geometry::Geometry* geo_ptr = entity.Geometry();
  Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);

  // Get basis basis functions
  const lf::uscalfe::FeLagrangeO1Tria<SCALAR> fe_space;
  const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> ref_grads =
      fe_space.GradientsReferenceShapeFunctions(Eigen::VectorXd::Zero(2))
          .transpose();
  const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> jacobian_inv =
      geo_ptr->JacobianInverseGramian(Eigen::VectorXd::Zero(2));

  // Compute gradients
  const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> grads =
      jacobian_inv * ref_grads;

  // Obtain reference shape functions
  const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> lambda =
      fe_space.EvalReferenceShapeFunctions(local_coords);

  // Take into account orientation
  auto edgeOrientations = entity.RelativeOrientations();

  // Obtain global coordinates
  std::vector<lf::base::size_type> global_indices(
      fe_space.NumRefShapeFunctions());
  auto edges = entity.SubEntities(1);
  for (lf::base::size_type iter = 0; iter < 3; ++iter) {
    global_indices[iter] = mesh_p_->Index(*edges[iter]);
  }

  // Compute the one form
  const int kShape = fe_space.NumRefShapeFunctions();
  for (int i = 0; i < local_coords.cols(); ++i) {
    Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> basis_func(
        grads.rows(), kShape);
    for (int j = 0; j < kShape; ++j) {
      basis_func.col(j) = lambda(j, i) * grads.col((j + 1) % kShape) -
                          lambda((j + 1) % kShape, i) * grads.col(j);
      // account for orientation
      basis_func.col(j) *= lf::mesh::to_sign(edgeOrientations[j]);
    }

    result[i] = mu_(global_indices[0]) * basis_func.col(0) +
                mu_(global_indices[1]) * basis_func.col(1) +
                mu_(global_indices[2]) * basis_func.col(2);
  }
  return result;
}

}  // namespace ecu_scheme::assemble

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_MESH_FUNCTION_ONE_FORM_H_
