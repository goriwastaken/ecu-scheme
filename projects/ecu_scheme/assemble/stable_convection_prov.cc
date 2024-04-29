
#include "stable_convection_prov.h"

namespace ecu_scheme::assemble {
lf::mesh::utils::CodimMeshDataSet<double> initMassesVert(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 2, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    for (const lf::mesh::Entity *corner : entity->SubEntities(2)) {
      masses(*corner) += area * 1.0 / 20.0;
    }
  }
  return masses;
}
lf::mesh::utils::CodimMeshDataSet<double> initMassesEdges(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 1, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    for (const lf::mesh::Entity *corner : entity->SubEntities(1)) {
      masses(*corner) += area * 2.0 / 15.0;
    }
  }
  return masses;
}
lf::mesh::utils::CodimMeshDataSet<double> initMassesCells(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 0, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    masses(*entity) = area * 9.0 / 20.0;
  }
  return masses;
}

}  // namespace ecu_scheme::assemble