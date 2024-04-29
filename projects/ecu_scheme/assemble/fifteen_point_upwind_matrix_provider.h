
#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_FIFTEEN_POINT_UPWIND_MATRIX_PROVIDER_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_FIFTEEN_POINT_UPWIND_MATRIX_PROVIDER_H_

#include <Eigen/Core>
#include <tuple>

#include "convection_upwind_matrix_provider.h"
#include "lf/fe/scalar_fe_space.h"
#include "lf/mesh/utils/utils.h"
#include "lf/uscalfe/uscalfe.h"

namespace ecu_scheme::assemble {

/**
 * @brief Prepare the 15-point quadrature rule for the reference triangle
 * @return a vector of pairs, where each pair contains a quadrature point and
 * its weight
 */
std::vector<std::tuple<Eigen::Vector2d, double, bool>>
prepareFifteenPointQuadRule();

lf::mesh::utils::CodimMeshDataSet<double> initMassesVerticesFifteenQuadRule(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p);

lf::mesh::utils::CodimMeshDataSet<double>
initMassesEdgeMidpointsFifteenQuadRule(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p);

lf::mesh::utils::CodimMeshDataSet<double> initMassesEdgeOffFifteenQuadRule(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p);

lf::mesh::utils::CodimMeshDataSet<double> initMassesCellsFifteenQuadRule(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p);

template <typename SCALAR, typename FUNCTOR>
class FifteenPointUpwindMatrixProvider {
 public:
  FifteenPointUpwindMatrixProvider(
      const std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>> &fe_space, FUNCTOR v,
      lf::mesh::utils::CodimMeshDataSet<double> masses_vertices,
      lf::mesh::utils::CodimMeshDataSet<double> masses_edge_midpoints,
      lf::mesh::utils::CodimMeshDataSet<double> masses_edges,
      lf::mesh::utils::CodimMeshDataSet<double> masses_cells);

  Eigen::Matrix<SCALAR, 6, 6> Eval(const lf::mesh::Entity &entity);

  bool isActive(const lf::mesh::Entity & /*entity*/) const { return true; }

 private:
  FUNCTOR v_;
  std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>> fe_space_;
  lf::mesh::utils::CodimMeshDataSet<double> masses_vertices_;
  lf::mesh::utils::CodimMeshDataSet<double> masses_edge_midpoints_;
  lf::mesh::utils::CodimMeshDataSet<double> masses_edges_;
  lf::mesh::utils::CodimMeshDataSet<double> masses_cells_;
};

template <typename SCALAR, typename FUNCTOR>
FifteenPointUpwindMatrixProvider<SCALAR, FUNCTOR>::
    FifteenPointUpwindMatrixProvider(
        const std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>> &fe_space,
        FUNCTOR v, lf::mesh::utils::CodimMeshDataSet<double> masses_vertices,
        lf::mesh::utils::CodimMeshDataSet<double> masses_edge_midpoints,
        lf::mesh::utils::CodimMeshDataSet<double> masses_edges,
        lf::mesh::utils::CodimMeshDataSet<double> masses_cells)
    : fe_space_(fe_space),
      v_(v),
      masses_vertices_(masses_vertices),
      masses_edge_midpoints_(masses_edge_midpoints),
      masses_edges_(masses_edges),
      masses_cells_(masses_cells) {}

template <typename SCALAR, typename FUNCTOR>
Eigen::Matrix<SCALAR, 6, 6>
FifteenPointUpwindMatrixProvider<SCALAR, FUNCTOR>::Eval(
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
  // Compute the fifteen quadrature points for exactness of degree p:
  // THIS IS FOR THE REFERENCE TRIANGLE
  //    14
  //   |    \
  //   12    13
  //   |        \
  //   9 - 10 -  11
  //   |            \
  //   5 - 6 - - 7 -  8
  //   |                 \
  //   0 -- 1 -- 2 -- 3 -- 4
  // They have the form (i/p, j/p) for i,j in {0,1,...,p} and i+j <= p
  Eigen::MatrixXd quadrature_points(2, 15);
  int temp_index = 0;  // temporary index to insert points into matrix
  const int kMaxDegreeOfExactness = 4;
  for (int i = 0; i <= kMaxDegreeOfExactness; ++i) {
    for (int j = 0; j <= kMaxDegreeOfExactness; ++j) {
      if (i + j > kMaxDegreeOfExactness) {
        continue;
      }
      quadrature_points.col(temp_index) =
          (Eigen::Vector2d(2, 1) << ((double)i / (double)kMaxDegreeOfExactness),
           ((double)j / (double)kMaxDegreeOfExactness))
              .finished();
      temp_index++;
    }
  }
  LF_ASSERT_MSG(temp_index == 15, "There must be 15 quadrature points");
  // In order to compare the obtained points we instantiate the precomputed
  // Quadrature rule for the reference triangle
  const std::vector<std::tuple<Eigen::Vector2d, double, bool>>
      quad_rule_reference = prepareFifteenPointQuadRule();

  // Prepare matrices for the computation of the element matrix and separate
  // implementation for linear and quadratic FE space
  Eigen::MatrixXd velocities(2, num_local_dofs);
  Eigen::MatrixXd all_nodes(2, num_local_dofs);
  std::vector<double> local_masses;

  all_nodes << corners, midpoints;
  velocities << v_(all_nodes.col(0)), v_(all_nodes.col(1)),
      v_(all_nodes.col(2)), v_(all_nodes.col(3)), v_(all_nodes.col(4)),
      v_(all_nodes.col(5));

  // Prepare a vector to store the 9 different quad points not available in
  // S_2^0(M)

  std::vector<Eigen::Vector2d> truncated_transformed_quad_points;
  // Transform the quadrature points from reference coords to physical
  // coordinates
  Eigen::MatrixXd transformed_quadrature_points =
      geo_ptr->Global(quadrature_points);
  // Remove the quadrature points that are associated with vertices and edge
  // midpoints We are interested only in the other 9 quad points
  Eigen::MatrixXd edge_off_points(2, 6);
  edge_off_points << transformed_quadrature_points.col(1),
      transformed_quadrature_points.col(3),
      transformed_quadrature_points.col(5),
      transformed_quadrature_points.col(8),
      transformed_quadrature_points.col(12),
      transformed_quadrature_points.col(13);
  Eigen::MatrixXd cell_center_points(2, 3);
  cell_center_points << transformed_quadrature_points.col(6),
      transformed_quadrature_points.col(7),
      transformed_quadrature_points.col(10);

  for (const lf::mesh::Entity *e : entity.SubEntities(2)) {
    local_masses.push_back(masses_vertices_(*e));
  }
  for (const lf::mesh::Entity *e : entity.SubEntities(1)) {
    local_masses.push_back(masses_edge_midpoints_(*e));
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

  // Contribution for edge quad points that are not midpoints of edges
  std::vector<double> local_masses_edge_off;
  for (const lf::mesh::Entity *e : entity.SubEntities(1)) {
    local_masses_edge_off.push_back(masses_edges_(*e));
  }
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
        LF_ASSERT_MSG(element_matrix(l, 0) == 0,
                      "Element matrix entry must be zero");
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
        element_matrix.row(l) = local_masses[l] * contribution;
      } else {
        // m^l is not upwind
        element_matrix.row(l) = Eigen::Vector<SCALAR, 6>::Zero();
      }
    }

    // Account for contribution of the other 15 - 6 = 9 quadrature points.
    // Contribution is added to each row of the element matrix.

    // Make a case distinction since LehrFEM++ edge numbering is a bit different
    // than the numbering from how the quad points are created
    int temp_local_index_edges = 0;
    for (auto off_edge : edge_off_points.colwise()) {
      Eigen::Matrix<double, 1, 6> off_edge_contribution =
          (v_(off_edge).transpose() * gradientsLocalShapeFunctions(off_edge))
              .cwiseProduct(localShapeFunctions(off_edge));
      // add upwind property because of edges
      const Eigen::Matrix<double, 2, 3> outward_normals_off =
          ecu_scheme::assemble::computeOutwardNormalsTria(entity);
      //      if(vl.dot(outward_normals_off.col(l%3)) < 0){
      //        // edge not upwind
      //        off_edge_contribution = Eigen::Matrix<double, 1, 6>::Zero();
      //      }else{
      if (temp_local_index_edges == 0 || temp_local_index_edges == 1) {
        off_edge_contribution *= local_masses_edge_off[0];
      } else if (temp_local_index_edges == 2 || temp_local_index_edges == 4) {
        off_edge_contribution *= local_masses_edge_off[1];
      } else if (temp_local_index_edges == 3 || temp_local_index_edges == 5) {
        off_edge_contribution *= local_masses_edge_off[2];
      } else {
        std::cerr << "There should be a total of 6 edge points that are not "
                     "edge midpoints"
                  << std::endl;
      }
      //      }
      element_matrix.row(l) += off_edge_contribution;
      temp_local_index_edges++;
    }
    for (auto cell_node : cell_center_points.colwise()) {
      Eigen::Matrix<double, 1, 6> cell_center_contribution =
          masses_cells_(entity) *
          (v_(cell_node).transpose() * gradientsLocalShapeFunctions(cell_node))
              .cwiseProduct(localShapeFunctions(cell_node));
      element_matrix.row(l) += cell_center_contribution;
    }
  }
  return element_matrix;
}

}  // namespace ecu_scheme::assemble

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_FIFTEEN_POINT_UPWIND_MATRIX_PROVIDER_H_
