
#include "rotating_hump_sol.h"

namespace ecu_scheme::experiments {

Eigen::Matrix<double, 2, 3> computeOutwardNormals(
    const lf::mesh::Entity& entity) {
  Eigen::Matrix<double, 2, 3> normals;
  const lf::geometry::Geometry* geo_ptr = entity.Geometry();
  const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);

  Eigen::Matrix<double, 3, 3> delta;
  delta.block<3, 2>(0, 0) = corners.transpose();
  delta.block<3, 1>(0, 2) = Eigen::Vector3d::Ones();
  const double det = delta.determinant();
  // rotation matrix of angle pi/2 is: 0, -1,
  //                                   1,  0
  const Eigen::Matrix<double, 2, 2> rotation_matrix =
      (Eigen::Matrix<double, 2, 2>() << 0, -1, 1, 0).finished();
  for (int i = 0; i < 3; ++i) {
    const Eigen::Vector2d edge = corners.col((i + 1) % 3) - corners.col(i);
    normals.col(i) = rotation_matrix * edge;
    normals.col(i) /= normals.col(i).norm();
  }
  if (det > 0) {
    // flip sign
    normals *= -1;
  }

  return normals;
}

Eigen::Matrix<double, 2, 3> computeTangentialComponents(
    const lf::mesh::Entity& entity) {
  const Eigen::Matrix<double, 2, 3> tria_normals =
      computeOutwardNormals(entity);
  Eigen::Matrix<double, 2, 3> tangential_components;
  tangential_components << -tria_normals.row(1), tria_normals.row(0);
  return tangential_components;
}

void EnforceInflowBDCOneform(
    const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>>& fe_space,
    lf::assemble::COOMatrix<double>& A, Eigen::VectorXd& phi,
    std::function<
        Eigen::Matrix<double, 2, 1>(const Eigen::Matrix<double, 2, 1, 0>&)>
        dirichlet) {
  // Generate DOFHandler object corresponding to edge element basis functions
  auto mesh_p = fe_space->Mesh();
  const lf::assemble::DofHandler& dofh_edge = lf::assemble::UniformFEDofHandler(
      mesh_p, {{lf::base::RefEl::kSegment(), 1}});
  lf::mesh::utils::AllCodimMeshDataSet<bool> bd_flags(mesh_p, false);
  // tolerance for geometric tests
  const double kTol = 1e-8;

  // Loop over all edges
  for (const auto& edge : mesh_p->Entities(1)) {
    LF_ASSERT_MSG(edge->RefEl() == lf::base::RefEl::kSegment(),
                  "Entity should be an edge");
    const lf::geometry::Geometry* geo_ptr = edge->Geometry();
    const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
    // Check if the edge lies on $\Gamma_{\mathrm{in}}$  (geometric test)
    if ((corners(0, 0) + corners(0, 1)) / 2. > 1. - kTol ||
        (corners(1, 0) + corners(1, 1)) / 2. < kTol) {
      // Add the edge to the flagged entities
      bd_flags(*edge) = true;
    }
  }
  // Set boundary condition data to flagged edges
  // remember initEssentialConditionFromFunction() doesn't work if underlying FE
  // space remains the lagrangian one
  //  auto selector = [&](lf::base::size_type idx) -> std::pair<bool, double> {
  //    const auto &e = dofh.Entity(idx);
  //    return {e.RefEl() == lf::base::RefElType::kSegment() && bd_flags(*e),
  //    0};
  //  };

  std::vector<std::pair<bool, double>> ess_dof_select(dofh_edge.NumDofs(),
                                                      {false, 0.0});
  // We want to enforce inflow BDC at tangential components of the edges
  // One way to do this is iterate through all cells and stop at marked edges
  // This allows to precompute the outward normals, and thus tangential
  // components beforehand
  for (const lf::mesh::Entity* entity : mesh_p->Entities(0)) {
    const Eigen::Matrix<double, 2, 3> tangential_components =
        computeTangentialComponents(*entity);
    unsigned int edge_idx = 0;
    for (const lf::mesh::Entity* edge : entity->SubEntities(1)) {
      // indexing is consistent with the one of the edge DOF handler
      const lf::geometry::Geometry* geo_ptr = edge->Geometry();
      const Eigen::MatrixXd endpoints = lf::geometry::Corners(*geo_ptr);
      if (bd_flags(*edge)) {
        const Eigen::Vector2d eval_bdc_at_midpoint =
            dirichlet((endpoints.col(0) + endpoints.col(1)) / 2.);
        const double tg_contribution =
            tangential_components.col(edge_idx).transpose() *
            eval_bdc_at_midpoint;

        auto gdof_indices{dofh_edge.GlobalDofIndices(*edge)};
        LF_ASSERT_MSG(
            gdof_indices.size() == 1,
            "Edge should have 1 DOF for edge element basis functions");
        // Set flags and values, NO accumulation!
        for (const lf::assemble::gdof_idx_t gdof_idx : gdof_indices) {
          ess_dof_select[gdof_idx] = {true, tg_contribution};
        }
      }
      edge_idx++;
    }
  }
  // Apply the boundary conditions to the linear system
  lf::assemble::FixFlaggedSolutionCompAlt<double>(
      [&ess_dof_select](lf::assemble::glb_idx_t dof_idx) {
        return ess_dof_select[dof_idx];
      },
      A, phi);
}

}  // namespace ecu_scheme::experiments