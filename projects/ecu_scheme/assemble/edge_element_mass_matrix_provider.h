
#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_EDGE_ELEMENT_MASS_MATRIX_PROVIDER_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_EDGE_ELEMENT_MASS_MATRIX_PROVIDER_H_

#include <Eigen/Core>

#include "convection_upwind_matrix_provider.h"
#include "lf/fe/scalar_fe_space.h"
#include "lf/mesh/utils/utils.h"
#include "lf/uscalfe/uscalfe.h"

namespace ecu_scheme::assemble {

/**
 * @brief Class providing the edge element mass matrix for a given FE space and
 * velocity field \f$\vec{\beta}\f$. The element matrix provider evaluates part
 * of the bilinear form for 1-forms from section 3.1, namely:
 * \f[
 * (\omega_h, \eta_h) \mapsto \int_T \mathcal{I}_{\vec{\beta},
 * 1}^1(\mathrm{i}_{\vec{\beta}} \mathrm{d}^1 \omega_h) \wedge \star \eta_h \f]
 * where the basis functions for \f$\omega_h, \ \eta_h \f$ are the edge element
 * basis functions defined as:
 * @f[
 * \mathbf{\psi}_i = s_i (\lambda_i \mathbf{grad}_{\Gamma}(\lambda_{i+1}) -
 * \lambda_{i+1} \mathbf{grad}_{\Gamma}(\lambda_{i}))
 * @f]
 *
 * @tparam SCALAR the scalar type of the FE space
 * @tparam FUNCTOR the type of the velocity field \f$\vec{\beta}\f$
 */
template <typename SCALAR, typename FUNCTOR>
class EdgeElementMassMatrixProvider {
 public:
  /**
   * @brief Constructor for the edge element mass matrix provider
   * @param fe_space the FE space
   * @param v the velocity field \f$\vec{\beta}\f$
   */
  EdgeElementMassMatrixProvider(
      const std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>>& fe_space,
      FUNCTOR v);
  /**
   * @brief Evaluates the edge element mass matrix for a given entity
   * @param entity reference to the entity
   * @return Edge element mass matrix
   */
  Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> Eval(
      const lf::mesh::Entity& entity) const;

  /** @brief Default implementation: all cells are active */
  bool isActive(const lf::mesh::Entity& /*entity*/) const { return true; }

 private:
  FUNCTOR v_;
  std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>> fe_space_;
};

template <typename SCALAR, typename FUNCTOR>
EdgeElementMassMatrixProvider<SCALAR, FUNCTOR>::EdgeElementMassMatrixProvider(
    const std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>>& fe_space, FUNCTOR v)
    : fe_space_(fe_space), v_(v) {}

template <typename SCALAR, typename FUNCTOR>
Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>
EdgeElementMassMatrixProvider<SCALAR, FUNCTOR>::Eval(
    const lf::mesh::Entity& entity) const {
  // Only triangles are supported
  LF_VERIFY_MSG(entity.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type" << entity.RefEl());

  // Get the geometry of the cell
  const lf::geometry::Geometry* geo_ptr = entity.Geometry();
  const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
  const double area = lf::geometry::Volume(*geo_ptr);
  LF_ASSERT_MSG(area > 0, "Area of cell must be positive");
  // Fetch edges of cell
  const nonstd::span<const lf::mesh::Entity* const> edges{
      entity.SubEntities(1)};

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
    Eigen::MatrixXd grads = j_inv_trans * reference_grads;
    // correct orientation
    auto edge_orientations = entity.RelativeOrientations();
    // obtain coefficient for edge elements
    Eigen::VectorXd s_coeff(3);
    s_coeff << lf::mesh::to_sign(edge_orientations[0]),
        lf::mesh::to_sign(edge_orientations[1]),
        lf::mesh::to_sign(edge_orientations[2]);

    // Compute element matrix for product of baricentric coord functions
    std::vector<Eigen::Matrix3d> elem_mat_bary(4);
    // clang-format off
    elem_mat_bary[0] << 2, 1, 1,
                        1, 2, 1,
                        1, 1, 2;
    elem_mat_bary[1] << 1, 1, 2,
                        2, 1, 1,
                        1, 2, 1;
    elem_mat_bary[1] *= -1;
    elem_mat_bary[2] << 1, 2, 1,
                        1, 1, 2,
                        2, 1, 1;
    elem_mat_bary[2] *= -1;
    elem_mat_bary[3] << 2, 1, 1,
                        1, 2, 1,
                        1, 1, 2;
    // clang-format on
    for (int i = 0; i < 4; ++i) {
      elem_mat_bary[i] *= area / 12.0;
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        elem_mat_bary[0](i, j) *=
            (grads.col((i + 1) % 3).transpose() * grads.col((j + 1) % 3));
        elem_mat_bary[0](i, j) *= s_coeff(i) * s_coeff(j);
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        elem_mat_bary[1](i, j) *=
            (grads.col((i + 1) % 3).transpose() * grads.col((j) % 3));
        elem_mat_bary[1](i, j) *= s_coeff(i) * s_coeff(j);
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        elem_mat_bary[2](i, j) *=
            (grads.col((i) % 3).transpose() * grads.col((j + 1) % 3));
        elem_mat_bary[2](i, j) *= s_coeff(i) * s_coeff(j);
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        elem_mat_bary[3](i, j) *=
            (grads.col((i) % 3).transpose() * grads.col((j) % 3));
        elem_mat_bary[3](i, j) *= s_coeff(i) * s_coeff(j);
      }
    }
    // sum up contributions
    for (int i = 0; i < 4; ++i) {
      element_matrix += elem_mat_bary[i];
    }
    LF_ASSERT_MSG(element_matrix.cols() == 3 && element_matrix.rows() == 3,
                  "Wrong size of element matrix");

    // account for contribution from term -<Ru div(R \psi_{e_j}), e_i>, where R
    // is the clockwise rotation by pi/2 \psi_{e_j} is the edge element basis
    // function and <.,.> denotes the scalar product We know that div(R
    // \psi_{e_j}) = s_j * area(T), the derivation can be found in the Thesis
    // "Extrusion-Contraction Upwind Schemes" Remember that the contribution is
    // taken into account only if Ru div(R \psi_{e_j}) is in the upwind triangle
    // with respect to edge e_i
    const auto clockwise_rotation = [](const Eigen::Vector2d& xh) {
      return (Eigen::Vector2d() << xh(1), -xh(0)).finished();
    };
    for (int i = 0; i < 3; ++i) {
      // Fetch edge entity and endpoints - for edge e_i
      const lf::geometry::Geometry& edge_geo_ptr_i{*(edges[i]->Geometry())};
      const Eigen::MatrixXd edge_endpoints_i{
          lf::geometry::Corners(edge_geo_ptr_i)};
      // Get direction vector of the edge - for edge e_i
      const Eigen::Vector2d edge_vector_i =
          edge_endpoints_i.col(1) - edge_endpoints_i.col(0);
      for (int j = 0; j < 3; ++j) {
        // Fetch edge entity and endpoints - for edge e_j
        const lf::geometry::Geometry& edge_geo_ptr_j{*(edges[j]->Geometry())};
        const Eigen::MatrixXd edge_endpoints_j{
            lf::geometry::Corners(edge_geo_ptr_j)};
        // Get direction vector of the edge - for edge e_j
        const Eigen::Vector2d edge_vector_j =
            edge_endpoints_j.col(1) - edge_endpoints_j.col(0);

        // Update each entry of the element matrix with the previously mentioned
        // contribution CAREFUL! Contribution has a negative sign to s_coeff(j),
        // because we look at -Ru * div(Rw)
        Eigen::Vector2d velocity_ej = v_(edge_vector_j);
        const Eigen::Vector2d kRotatedVeloWithDivCoeff =
            clockwise_rotation(velocity_ej) * (-s_coeff(j) / area);
        const double contribution = kRotatedVeloWithDivCoeff.dot(edge_vector_i);

        // Check if the edge element basis function is in the upwind triangle
        // with respect to edge e_i
        const Eigen::Matrix<double, 2, 3> outward_edge_normals =
            ecu_scheme::assemble::computeOutwardNormalsTria(entity);
        if (velocity_ej.dot(outward_edge_normals.col(i)) >= 0) {
          // we are in the upwind triangle with respect to e_i
          if (velocity_ej.dot(outward_edge_normals.col(i)) == 0) {
            // edge element basis function is on the edge e_i
            // contribution is halved
            element_matrix(i, j) *= 0.5 * contribution;
          } else {
            element_matrix(i, j) *= contribution;
          }
        } else {
          // not upwind - contribution is set to 0
          element_matrix(i, j) = 0;
        }
      }  // end loop over j
    }    // end loop over i

    return element_matrix;
  }  // end linear FE case
  else {
    // quadratic FE space
    // todo

    return element_matrix;
  }  // end quadratic FE case
}

}  // namespace ecu_scheme::assemble

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_EDGE_ELEMENT_MASS_MATRIX_PROVIDER_H_
