//
// 
//

#include "convection_upwind_matrix_provider.h"
#include <Eigen/LU>
namespace ecu_scheme::assemble {

/**
 * @brief Computes the masses m(p) of all vertices of the mesh in the linear FE space case
 * @param mesh_p pointer to the mesh
 * @return Data structure containing the masses m(p) for all vertices of the mesh represented by mesh_p.
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMasses(const std::shared_ptr<const lf::mesh::Mesh>& mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 2, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    for (const lf::mesh::Entity *corner : entity->SubEntities(2)) {
      masses(*corner) += area * 1.0/3.0;
    }
  }
  return masses;
}
/**
 * @brief Computes the masses m(p) of all vertices of the mesh in the quadratic FE space case
 * @param mesh_p pointer to the mesh
 * @return Data structure containing the masses m(p) for all vertices of the mesh represented by mesh_p.
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMassesQuadratic(const std::shared_ptr<const lf::mesh::Mesh>& mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 2, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    for (const lf::mesh::Entity *corner : entity->SubEntities(2)) {
      masses(*corner) += 0.0;
    }
  }
  return masses;
}
/**
 * @brief Computes the masses m(p) of all midpoints of edges of the mesh in the quadratic FE space case
 * @param mesh_p pointer to the mesh
 * @return Data structure containing the masses m(p) for all midpoints of edges of the mesh represented by mesh_p.
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMassesQuadraticEdges(const std::shared_ptr<const lf::mesh::Mesh>& mesh_p) {
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 1, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    for (const lf::mesh::Entity *corner : entity->SubEntities(1)) {
      masses(*corner) += area * 1.0/3.0;
    }
  }
  return masses;
}
/**
 * @brief Computes the outward normals of a triangular cell
 * @param entity the triangular cell
 * @return Matrix containing the outward normals of the triangular cell
 */
Eigen::Matrix<double, 2, 3> computeOutwardNormalsTria(const lf::mesh::Entity &entity) {
  Eigen::Matrix<double, 2, 3> normals;
  const lf::geometry::Geometry *geo_ptr = entity.Geometry();
  const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);

  Eigen::Matrix<double, 3, 3> delta;
  delta.block<3,2>(0,0) = corners.transpose();
  delta.block<3,1>(0,2) = Eigen::Vector3d::Ones();
  const double det = delta.determinant();
  //rotation matrix of angle pi/2 is: 0, -1,
  //                                  1,  0
  const Eigen::Matrix<double,2,2> rotation_matrix = (Eigen::Matrix<double,2,2>() << 0, -1, 1, 0).finished();
  for(int i = 0; i < 3; ++i){
    const Eigen::Vector2d edge = corners.col((i+1)%3) - corners.col(i);
    normals.col(i) = rotation_matrix * edge;
    normals.col(i) /= normals.col(i).norm();
  }
  if(det > 0){
    // flip sign
    normals *= -1;
  }

  return normals;
}


} // assemble