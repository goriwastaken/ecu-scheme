#include "concentric_stream_solution.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <memory>

namespace ecu_scheme::experiments {

lf::mesh::utils::CodimMeshDataSet<bool> flagNodesOnInflowBoundary(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p,
    std::function<
        Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double, 2, 1, 0> &)>
        velocity) {
  // wrap velocity functor into a LehrFEM Meshfunction
  lf::mesh::utils::MeshFunctionGlobal<decltype(velocity)> mf_velocity{velocity};
  // Data set of flags
  lf::mesh::utils::CodimMeshDataSet<bool> bd_inflow_flags(mesh_p, 2, false);
  // Reference coordinate for the barycenter of the reference triangle
  const Eigen::MatrixXd bary_hat = Eigen::Vector2d(1.0 / 3.0, 1.0 / 3.0);
  // Reference coordinates for midpoints of edges
  const Eigen::MatrixXd midpoints_hat =
      (Eigen::Matrix<double, 2, 3>() << 0.5, 0.5, 0.0, 0.0, 0.5, 0.5)
          .finished();
  // Mark edges (codim = 1) on the boundary
  lf::mesh::utils::CodimMeshDataSet<bool> edge_bd_flags(
      lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1));
  // Loop over the cells of the mesh
  for (const lf::mesh::Entity *cell : mesh_p->Entities(0)) {
    // Fetch geometry of cell
    const lf::geometry::Geometry *geo_ptr = cell->Geometry();
    LF_ASSERT_MSG(cell->RefEl() == lf::base::RefEl::kTria(),
                  "Only implemented for triangular cells");
    LF_ASSERT_MSG(geo_ptr->DimGlobal() == 2, "Only implemented for 2D cells");
    // Fetch the global coordinates of the barycenter of the triangle
    const Eigen::Vector2d physical_barycenter{geo_ptr->Global(bary_hat).col(0)};
    // Compute velocity in reference midpoints
    auto velocity_ref_midpoint_values = mf_velocity(*cell, midpoints_hat);
    // Retrieve pointers to all edges of the triangle
    nonstd::span<const lf::mesh::Entity *const> edges{cell->SubEntities(1)};
    LF_ASSERT_MSG(edges.size() == 3, "Triangle should have 3 edges");
    for (int i = 0; i < 3; ++i) {
      if (edge_bd_flags(*edges[i])) {
        const lf::geometry::Geometry *edge_geo_ptr = edges[i]->Geometry();
        const Eigen::Matrix edge_points = lf::geometry::Corners(*edge_geo_ptr);
        // Direction vector of the edge
        const Eigen::Vector2d direction =
            edge_points.col(1) - edge_points.col(0);
        // Rotate counterclockwise by 90 degrees
        const Eigen::Vector2d edge_normal =
            Eigen::Vector2d(direction(1), -direction(0));
        // Adjust orientation s.t. we have an outward pointing normal
        const int orientation =
            (edge_normal.dot(physical_barycenter - edge_points.col(0)) > 0) ? -1
                                                                            : 1;
        // Check angle of exterior normal and velocity vector
        const int v_rel_orientation =
            ((velocity_ref_midpoint_values[i].dot(edge_normal) > 0) ? 1 : -1) *
            orientation;
        if (v_rel_orientation < 0) {
          // Inflow boundary: obtain endpoints of the edge and mark them
          nonstd::span<const lf::mesh::Entity *const> endpoints{
              edges[i]->SubEntities(1)};
          LF_ASSERT_MSG(endpoints.size() == 2, "Edge must have 2 endpoints");
          bd_inflow_flags(*endpoints[0]) = true;
          bd_inflow_flags(*endpoints[1]) = true;
        }
      }
    }
  }
  return bd_inflow_flags;
}

}  // namespace ecu_scheme::experiments
