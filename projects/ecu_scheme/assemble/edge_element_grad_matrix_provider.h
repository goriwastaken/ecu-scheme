
#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_EDGE_ELEMENT_GRAD_MATRIX_PROVIDER_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_EDGE_ELEMENT_GRAD_MATRIX_PROVIDER_H_

#include <Eigen/Core>

#include "lf/fe/scalar_fe_space.h"
#include "lf/mesh/utils/utils.h"
#include "lf/uscalfe/uscalfe.h"

namespace ecu_scheme::assemble {

/**
 * @brief Class providing the edge element gradient matrix for a given FE space
 * and velocity field
 * * The element matrix provider works in a 3 dimensional world with 2
 * dimensional triangular cells. And evaluates the bilinear form
 * \f[
 * (u, \mathbf{v}) \mapsto \int\limits_K \mathbf{grad}_{\Gamma}(u) \, \mathbf{v}
 * \ dx
 * \f]
 *
 * Basis functions are the Whitney 1-forms, surface edge elements for
 * @f$\mathbf{v}@f$ and the barycentric basis functions for @f$u@f$
 *
 * The whitney 1-forms, surface edge elements are associated with
 * edges and defined as
 *
 * @f[
 *  \mathbf{b}_i = s_i (\lambda_i \mathbf{grad}_{\Gamma}(\lambda_{i+1}) -
 * \lambda_{i+1} \mathbf{grad}_{\Gamma}(\lambda_{i}))
 * @f]
 * @tparam SCALAR
 * @tparam FUNCTOR
 */
template <typename SCALAR, typename FUNCTOR>
class EdgeElementGradMatrixProvider {
 public:
  EdgeElementGradMatrixProvider(
      const std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>>& fe_space,
      FUNCTOR v);

  Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> Eval(
      const lf::mesh::Entity& entity) const;

  bool isActive(const lf::mesh::Entity& /*entity*/) const { return true; }

 private:
  FUNCTOR v_;
  std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>> fe_space_;
};

template <typename SCALAR, typename FUNCTOR>
EdgeElementGradMatrixProvider<SCALAR, FUNCTOR>::EdgeElementGradMatrixProvider(
    const std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>>& fe_space, FUNCTOR v)
    : fe_space_(fe_space), v_(v) {}

template <typename SCALAR, typename FUNCTOR>
Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>
EdgeElementGradMatrixProvider<SCALAR, FUNCTOR>::Eval(
    const lf::mesh::Entity& entity) const {
  // Only triangles are supported
  LF_VERIFY_MSG(entity.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type" << entity.RefEl());

  // Get the geometry of the cell
  const lf::geometry::Geometry* geo_ptr = entity.Geometry();
  const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
  const double area = lf::geometry::Volume(*geo_ptr);
  LF_ASSERT_MSG(area > 0, "Area of cell must be positive");

  // Compute signed area of the cell for checking the orientation of vertices
  const double signed_area =
      (corners(0, 1) - corners(0, 0)) * (corners(1, 2) - corners(1, 0)) -
      (corners(0, 2) - corners(0, 0)) * (corners(1, 1) - corners(1, 0));
  // Set orientation based on signed area
  const bool is_clockwise = (signed_area < 0);

  // Compute normals of vertices of the triangular cell
  Eigen::Matrix3d X;
  X.col(0) = Eigen::Vector3d::Ones();
  X.rightCols(2) = corners.transpose();
  const Eigen::MatrixXd temporary_gradients = X.inverse().bottomRows(2);
  const Eigen::MatrixXd vertex_normals = -temporary_gradients;

  const size_t num_local_dofs = fe_space_->LocGlobMap().NumLocalDofs(entity);
  LF_ASSERT_MSG(num_local_dofs == 6 || num_local_dofs == 3,
                "Only quadratic or linear FE spaces are supported");
  Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> element_matrix(
      num_local_dofs, num_local_dofs);
  element_matrix.setZero();

  if (num_local_dofs == 3) {
    // linear FE space
    const lf::uscalfe::FeLagrangeO1Tria<SCALAR> hat_function;
    // construct basis functions from curl of reference hat functions
    const Eigen::MatrixXd reference_grads =
        hat_function.GradientsReferenceShapeFunctions(Eigen::VectorXd::Zero(2))
            .transpose();
    // compute jacobian inverse gramian
    const Eigen::MatrixXd j_inv_trans =
        geo_ptr->JacobianInverseGramian(Eigen::VectorXd::Zero(2));
    // compute gradients
    const Eigen::MatrixXd grads = j_inv_trans * reference_grads;

    // edge element construction
    Eigen::MatrixXd edge_element(grads.rows(), 3);
    for (int i = 0; i < 3; ++i) {
      edge_element.col(i) = grads.col((i + 1) % 3) - grads.col(i);
    }
    // correct orientation
    auto edge_orientations = entity.RelativeOrientations();
    for (int i = 0; i < 3; ++i) {
      edge_element.col(i) *= lf::mesh::to_sign(edge_orientations[i]);
    }
    // compute part of the element matrix -- what is given by \int_T \nabla
    // \lambda_i \psi_{e_j}
    const Eigen::Matrix3d contribution_element_matrix =
        (area / 3.0) * edge_element.transpose() * grads;

    // Row-vector of barycentric coordinate functions based on global
    // coordinates of nodes
    auto bary_functions =
        [area, corners, is_clockwise](
            const Eigen::Vector2d& xh) -> Eigen::Matrix<double, 1, 3> {
      Eigen::Matrix<double, 1, 3> bary;
      const double coeff = 1.0 / (2.0 * area);
      // barycentric function 1
      bary.col(0) = coeff * (xh - corners.col(1)).transpose() *
                    (Eigen::Vector2d(2, 1) << corners(1, 1) - corners(1, 2),
                     corners(0, 2) - corners(0, 1))
                        .finished();
      // bary 2
      bary.col(1) = coeff * (xh - corners.col(2)).transpose() *
                    (Eigen::Vector2d(2, 1) << corners(1, 2) - corners(1, 0),
                     corners(0, 0) - corners(0, 2))
                        .finished();
      // bary 3
      bary.col(2) = coeff * (xh - corners.col(0)).transpose() *
                    (Eigen::Vector2d(2, 1) << corners(1, 0) - corners(1, 1),
                     corners(0, 1) - corners(0, 0))
                        .finished();
      if (is_clockwise) {
        return -bary;
      }
      return bary;
    };

    // Matrix of gradients of barycentric coordinate functions based on global
    // coordinates of nodes
    auto bary_functions_grad =
        [area, corners, is_clockwise](
            const Eigen::Vector2d& xh) -> Eigen::Matrix<double, 2, 3> {
      Eigen::Matrix<double, 2, 3> bary_grad;
      const double coeff = 1.0 / (2.0 * area);
      // bary 1
      bary_grad.col(0) =
          coeff * (Eigen::Vector2d(2, 1) << corners(1, 1) - corners(1, 2),
                   corners(0, 2) - corners(0, 1))
                      .finished();
      // bary 2
      bary_grad.col(1) =
          coeff * (Eigen::Vector2d(2, 1) << corners(1, 2) - corners(1, 0),
                   corners(0, 0) - corners(0, 2))
                      .finished();
      // bary 3
      bary_grad.col(2) =
          coeff * (Eigen::Vector2d(2, 1) << corners(1, 0) - corners(1, 1),
                   corners(0, 1) - corners(0, 0))
                      .finished();
      if (is_clockwise) {
        return -bary_grad;
      }
      return bary_grad;
    };
    // compute psi_{e_i}(x_i)
    auto edge_elem_basis = [edge_orientations, bary_functions,
                            bary_functions_grad](const Eigen::Vector2d& x) {
      Eigen::Matrix<double, 2, 3> edge_elem_basis;
      Eigen::Matrix<double, 1, 3> basis_lambda = bary_functions(x);
      Eigen::Matrix<double, 2, 3> basis_lambda_grad = bary_functions_grad(x);
      edge_elem_basis.col(0) = basis_lambda_grad.col(1) * basis_lambda.col(0) -
                               basis_lambda_grad.col(0) * basis_lambda.col(1);
      edge_elem_basis.col(1) = basis_lambda_grad.col(2) * basis_lambda.col(1) -
                               basis_lambda_grad.col(1) * basis_lambda.col(2);
      edge_elem_basis.col(2) = basis_lambda_grad.col(0) * basis_lambda.col(2) -
                               basis_lambda_grad.col(2) * basis_lambda.col(0);

      // correct orientation
      edge_elem_basis.col(0) *= lf::mesh::to_sign(edge_orientations[0]);
      edge_elem_basis.col(1) *= lf::mesh::to_sign(edge_orientations[1]);
      edge_elem_basis.col(2) *= lf::mesh::to_sign(edge_orientations[2]);

      return edge_elem_basis;
    };
    // account for contribution that is row-wise constant -- (u \cdot
    // psi_{e_i})(x_i) contribution is 0 iff the vertex x_i is not part of the
    // upwind triangle
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector2d velocity_at_vertex = v_(corners.col(i));
      // compute contribution
      double contribution =
          velocity_at_vertex.transpose() *
          edge_elem_basis(
              (Eigen::Vector2d() << corners(0, i), corners(1, i)).finished())
              .col(i);

      if (velocity_at_vertex.dot(vertex_normals.col((i + 2) % 3)) >= 0 &&
          velocity_at_vertex.dot(vertex_normals.col((i + 1) % 3)) >= 0) {
        // vertex is part of the upwind triangle
        element_matrix += contribution * contribution_element_matrix;
      } else {
        // vertex is not part of the upwind triangle
        element_matrix += Eigen::Matrix3d::Zero();
      }
    }

    return element_matrix;
  }  // end linear FE case
  else {
    // quadratic FE space
    // todo
    return element_matrix;
  }  // end quadratic FE case
}

}  // namespace ecu_scheme::assemble

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_EDGE_ELEMENT_GRAD_MATRIX_PROVIDER_H_
