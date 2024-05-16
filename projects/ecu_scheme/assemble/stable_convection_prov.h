
#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_STABLE_CONVECTION_PROV_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_STABLE_CONVECTION_PROV_H_

#include <Eigen/Core>

#include "convection_upwind_matrix_provider.h"
#include "lf/fe/scalar_fe_space.h"
#include "lf/mesh/utils/utils.h"
#include "lf/uscalfe/uscalfe.h"

namespace ecu_scheme::assemble {

/**
 * @brief Computes the masses m(p) for the \f$b_h^{vmb}\f$ scheme of all
 * vertices of the mesh
 * @param mesh_p underlying mesh
 * @return Data structure containing the masses m(p) for all vertices of the
 * mesh
 */
lf::mesh::utils::CodimMeshDataSet<double> initMassesVert(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p);

/**
 * @brief Computes the masses m(p) for the \f$b_h^{vmb}\f$ scheme of all
 * midpoints of edges of the mesh
 * @param mesh_p underlying mesh
 * @return Data structure containing the masses m(p) for all midpoints of edges
 * of the mesh
 */
lf::mesh::utils::CodimMeshDataSet<double> initMassesEdges(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p);

/**
 * @brief Computes the masses m(p) for the \f$b_h^{vmb}\f$ scheme of all cells
 * of the mesh
 * @param mesh_p underlying mesh
 * @return Data structure containing the masses m(p) for all cells of the mesh
 */
lf::mesh::utils::CodimMeshDataSet<double> initMassesCells(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p);

/**
 * @brief Class providing the element matrix for the 7-point upwind scheme
 * The element matrix provider evaluates the convective bilinear form for the
 * 7-point upwind scheme \f$ b_h^{vmb}(u_h,v_h) \f$ presented in section 2.1.1
 * of the thesis
 * @tparam SCALAR the scalar type of the FE space
 * @tparam FUNCTOR the type of the velocity field \f$\vec{\beta}\f$
 */
template <typename SCALAR, typename FUNCTOR>
class StableConvectionUpwindMatrixProvider {
 public:
  /**
   * @brief Constructor for the 7-point scheme element matrix provider
   * @param fe_space underlying FE space
   * @param v velocity field \f$\vec{\beta}\f$
   * @param masses_vertices Data structure containing the masses m(p) for all
   * vertices of the mesh
   * @param masses_edges Data structure containing the masses m(p) for all
   * midpoints of edges of the mesh
   * @param masses_cells Data structure containing the masses m(p) for all cells
   * of the mesh
   */
  StableConvectionUpwindMatrixProvider(
      const std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>> &fe_space, FUNCTOR v,
      lf::mesh::utils::CodimMeshDataSet<double> masses_vertices,
      lf::mesh::utils::CodimMeshDataSet<double> masses_edges,
      lf::mesh::utils::CodimMeshDataSet<double> masses_cells);

  /**
   * @brief Evaluates the element matrix for a given entity
   * @param entity underlying entity
   * @return Element matrix
   */
  Eigen::Matrix<SCALAR, 6, 6> Eval(const lf::mesh::Entity &entity);

  /** @brief Default implementation: all cells are active */
  bool isActive(const lf::mesh::Entity & /*entity*/) const { return true; }

 private:
  FUNCTOR v_;
  std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>> fe_space_;
  lf::mesh::utils::CodimMeshDataSet<double> masses_vertices_;
  lf::mesh::utils::CodimMeshDataSet<double> masses_edges_;
  lf::mesh::utils::CodimMeshDataSet<double> masses_cells_;
};

template <typename SCALAR, typename FUNCTOR>
StableConvectionUpwindMatrixProvider<SCALAR, FUNCTOR>::
    StableConvectionUpwindMatrixProvider(
        const std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>> &fe_space,
        FUNCTOR v, lf::mesh::utils::CodimMeshDataSet<double> masses_vertices,
        lf::mesh::utils::CodimMeshDataSet<double> masses_edges,
        lf::mesh::utils::CodimMeshDataSet<double> masses_cells)
    : fe_space_(fe_space),
      v_(v),
      masses_vertices_(masses_vertices),
      masses_edges_(masses_edges),
      masses_cells_(masses_cells) {}

template <typename SCALAR, typename FUNCTOR>
Eigen::Matrix<SCALAR, 6, 6>
StableConvectionUpwindMatrixProvider<SCALAR, FUNCTOR>::Eval(
    const lf::mesh::Entity &entity) {
  const lf::geometry::Geometry *geo_ptr = entity.Geometry();
  const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
  const double area = lf::geometry::Volume(*geo_ptr);
  LF_ASSERT_MSG(area > 0, "Area of cell must be positive");

  const size_t num_local_dofs = fe_space_->LocGlobMap().NumLocalDofs(entity);
  LF_ASSERT_MSG(num_local_dofs == 6, "Only quadratic FE spaces are supported");
  Eigen::Matrix<SCALAR, 6, 6> element_matrix =
      Eigen::Matrix<SCALAR, 6, 6>::Zero();
  // Compute normals of vertices of the triangular cell
  Eigen::Matrix3d X;
  X.col(0) = Eigen::Vector3d::Ones();
  X.rightCols(2) = corners.transpose();
  Eigen::MatrixXd temporary_gradients = X.inverse().bottomRows(2);
  Eigen::MatrixXd vertex_normals = -temporary_gradients;

  // Compute signed area of the cell for checking the orientation of vertices
  const double signed_area =
      (corners(0, 1) - corners(0, 0)) * (corners(1, 2) - corners(1, 0)) -
      (corners(0, 2) - corners(0, 0)) * (corners(1, 1) - corners(1, 0));
  // Set orientation based on signed area
  const bool is_clockwise = (signed_area < 0);

  // Compute edge midpoint coordinates - only needed for quadratic FE space
  Eigen::MatrixXd midpoints(2, 3);
  for (int i = 0; i < 3; ++i) {
    midpoints.col(i) = 0.5 * (corners.col((i + 1) % 3) + corners.col(i));
  }
  // Compute barycenter of entity
  Eigen::Vector2d barycenter = corners.rowwise().mean();
  Eigen::Vector2d test_bary =
      (corners.col(0) + corners.col(1) + corners.col(2)) / 3.0;
  LF_ASSERT_MSG(barycenter == test_bary, "Barycenter computation failed");

  // Prepare matrices for the computation of the element matrix and separate
  // implementation for linear and quadratic FE space
  Eigen::MatrixXd velocities(2, num_local_dofs);
  Eigen::MatrixXd all_nodes(2, num_local_dofs);
  std::vector<double> local_masses;

  all_nodes << corners, midpoints;
  velocities << v_(all_nodes.col(0)), v_(all_nodes.col(1)),
      v_(all_nodes.col(2)), v_(all_nodes.col(3)), v_(all_nodes.col(4)),
      v_(all_nodes.col(5));

  for (const lf::mesh::Entity *e : entity.SubEntities(2)) {
    local_masses.push_back(masses_vertices_(*e));
  }
  for (const lf::mesh::Entity *e : entity.SubEntities(1)) {
    local_masses.push_back(masses_edges_(*e));
  }

  LF_ASSERT_MSG(local_masses.size() == 6,
                "There must be six masses, one for each basis function");

  // Row-vector of barycentric coordinate functions based on global coordinates
  // of nodes
  auto bary_functions =
      [area, corners,
       is_clockwise](const Eigen::Vector2d &xh) -> Eigen::Matrix<double, 1, 3> {
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
      [area, corners,
       is_clockwise](const Eigen::Vector2d &xh) -> Eigen::Matrix<double, 2, 3> {
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

  auto localShapeFunctions =
      [bary_functions](
          const Eigen::Vector2d xh) -> Eigen::Matrix<double, 1, 6> {
    Eigen::Matrix<double, 1, 6> shapeFunctions;
    Eigen::Matrix<double, 1, 3> temp = bary_functions(xh);
    shapeFunctions << temp(0, 0) * (2 * temp(0, 0) - 1),
        temp(0, 1) * (2 * temp(0, 1) - 1), temp(0, 2) * (2 * temp(0, 2) - 1),
        4 * temp(0, 0) * temp(0, 1), 4 * temp(0, 1) * temp(0, 2),
        4 * temp(0, 0) * temp(0, 2);
    return shapeFunctions;
  };

  // Matrix of gradients of local shape functions based on global coordinates of
  // nodes in quadratic Lagrangian FE space
  auto gradientsLocalShapeFunctions =
      [bary_functions, bary_functions_grad](
          const Eigen::Vector2d xh) -> Eigen::Matrix<double, 2, 6> {
    Eigen::Matrix<double, 2, 6> gradients;
    // barycentric coordinate functions
    Eigen::Matrix<double, 1, 3> temp = bary_functions(xh);
    //        Eigen::Matrix<double, 1, 3> temp = bary_new(xh);
    Eigen::Matrix<double, 2, 3> grads_bary = bary_functions_grad(xh);
    Eigen::RowVector3d l;
    l << temp(0, 0), temp(0, 1), temp(0, 2);

    gradients.col(0) = grads_bary.col(0) * (4 * l[0] - 1);
    gradients.col(1) = grads_bary.col(1) * (4 * l[1] - 1);
    gradients.col(2) = grads_bary.col(2) * (4 * l[2] - 1);
    gradients.col(3) =
        4 * (grads_bary.col(0) * l[1] + grads_bary.col(1) * l[0]);
    gradients.col(4) =
        4 * (grads_bary.col(1) * l[2] + grads_bary.col(2) * l[1]);
    gradients.col(5) =
        4 * (grads_bary.col(0) * l[2] + grads_bary.col(2) * l[0]);
    return gradients;
  };

  // Compute the element matrix
  for (int l = 0; l < 6; ++l) {
    Eigen::Vector2d vl = velocities.col(l);
    // compute product of v(a^l) with gradients of basis function of a^l
    Eigen::Matrix<double, 2, 6> grads =
        gradientsLocalShapeFunctions(all_nodes.col(l));
    Eigen::Matrix<double, 1, 6> contribution = vl.transpose() * grads;

    // Vertex a^l is upwind iff product of v(a^l) with both adjacent normals is
    // positive
    if (l < 3) {
      // first 3 nodes are vertices of triangle
      if (vl.dot(vertex_normals.col((l + 2) % 3)) >= 0 &&
          vl.dot(vertex_normals.col((l + 1) % 3)) >= 0) {
        // a^l is upwind
        element_matrix.row(l) = local_masses[l] * contribution;
        //        LF_ASSERT_MSG(element_matrix(l, 0) == 0, "Element matrix entry
        //        must be zero");
      } else {
        // a^l is not upwind
        element_matrix.row(l) = Eigen::Vector<SCALAR, 6>::Zero();
      }
    } else {
      // last 3 nodes are edge midpoints of triangle
      const Eigen::Matrix<double, 2, 3> outward_normals =
          ecu_scheme::assemble::computeOutwardNormalsTria(entity);
      // Midpoint m^l is upwind iff product of v(m^l) with corresponding outward
      // normals is positive
      if (vl.dot(outward_normals.col(l % 3)) >= 0) {
        // m^l is upwind
        if (vl.dot(outward_normals.col(l % 3)) == 0) {
          element_matrix.row(l) = 0.5 * local_masses[l] * contribution;
        } else {
          element_matrix.row(l) = local_masses[l] * contribution;
        }
      } else {
        // m^l is not upwind
        element_matrix.row(l) = Eigen::Vector<SCALAR, 6>::Zero();
      }
    }

    // Account for barycentric contribution for each row of the element matrix
    Eigen::Matrix<double, 1, 6> barycenter_row_contribution =
        masses_cells_(entity) *
        (v_(barycenter).transpose() * gradientsLocalShapeFunctions(barycenter))
            .cwiseProduct(localShapeFunctions(barycenter));
    element_matrix.row(l) += barycenter_row_contribution;
  }

  return element_matrix;
}

}  // namespace ecu_scheme::assemble

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_STABLE_CONVECTION_PROV_H_
