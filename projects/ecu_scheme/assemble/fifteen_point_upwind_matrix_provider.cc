
#include "fifteen_point_upwind_matrix_provider.h"

#include <tuple>

namespace ecu_scheme::assemble {
std::vector<std::tuple<Eigen::Vector2d, double, bool>>
prepareFifteenPointQuadRule() {
  std::vector<std::tuple<Eigen::Vector2d, double, bool>> quad_rule_result;
  // these points are with respect to the reference triangle
  // weights are computed with Mathematica to give a fifteen-point quadrature
  // rule exact for degree 4 polynomials
  std::vector<double> weights = {0.0,      4. / 45.,  -1.0 / 45., 4. / 45.,
                                 0.0,      4. / 45.,  8. / 45.,   8. / 45.,
                                 4. / 45., -1. / 45., 8. / 45.,   -1. / 45.,
                                 4. / 45., 4. / 45.,  0.0};
  // flag to indicate the vertices and edge midpoints that don't correspond to
  // S_2^0(M) nodes. different nodes are marked as true
  std::vector<bool> flags_different_nodes = {false, true,  false, true, false,
                                             true,  true,  true,  true, false,
                                             true,  false, true,  true, false};
  const int kMaxDegreeOfExactness = 4;
  int temporary_index = 0;
  for (int i = 0; i <= kMaxDegreeOfExactness; ++i) {
    for (int j = 0; j <= kMaxDegreeOfExactness; ++j) {
      if (i + j > kMaxDegreeOfExactness) {
        continue;
      }
      Eigen::Vector2d point;
      point << (static_cast<double>(i) /
                static_cast<double>(kMaxDegreeOfExactness)),
          (static_cast<double>(j) / static_cast<double>(kMaxDegreeOfExactness));
      std::tuple point_weight_flag =
          std::make_tuple(point, weights.at(temporary_index),
                          flags_different_nodes.at(temporary_index));
      quad_rule_result.emplace_back(point_weight_flag);
      temporary_index++;
    }
  }
  LF_ASSERT_MSG(temporary_index == 15,
                "Temporary index must be 15 at the end of insertions");

  return quad_rule_result;
}

lf::mesh::utils::CodimMeshDataSet<double> initMassesVerticesFifteenQuadRule(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 2, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    for (const lf::mesh::Entity *corner : entity->SubEntities(2)) {
      masses(*corner) += area * 0.0;
    }
  }
  return masses;
}

lf::mesh::utils::CodimMeshDataSet<double>
initMassesEdgeMidpointsFifteenQuadRule(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 1, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    for (const lf::mesh::Entity *corner : entity->SubEntities(1)) {
      masses(*corner) += area * (-1.0 / 45.0);
    }
  }
  return masses;
}

lf::mesh::utils::CodimMeshDataSet<double> initMassesEdgeOffFifteenQuadRule(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 1, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    for (const lf::mesh::Entity *corner : entity->SubEntities(1)) {
      masses(*corner) += area * (4.0 / 45.0);
    }
  }
  return masses;
}

lf::mesh::utils::CodimMeshDataSet<double> initMassesCellsFifteenQuadRule(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 0, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);

    masses(*entity) += area * (8.0 / 45.0);
  }
  return masses;
}

}  // namespace ecu_scheme::assemble