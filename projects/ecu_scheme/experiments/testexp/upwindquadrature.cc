/**
 * @file upwindquadrature.cc
 * @brief NPDE homework template
 * @author Philippe Peter
 * @date June 2020
 * @copyright Developed at SAM, ETH Zurich
 */

#include "upwindquadrature.h"
#include <lf/fe/fe.h>
#include <lf/assemble/assemble.h>

namespace UpwindQuadrature {

/**
 * @brief Computes the masses m(p) of all vertices of the mesh
 * @param mesh_p pointer to the mesh.
 * @return Datastructure containing the masses m(p) for all vertices p of the
 * mesh represented by mesh_p.
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMasses(
    std::shared_ptr<const lf::mesh::Mesh> mesh_p) {
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

}  // namespace UpwindQuadrature

namespace UpwindQuadratureQuadratic{
/**
 * @brief Computes the masses m(p) of all vertices of the mesh
 * @param mesh_p pointer to the mesh
 * @return Datastructure containing the masses m(p) for all vertices of the mesh represented by mesh_p.
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMassesQuadratic(std::shared_ptr<const lf::mesh::Mesh> mesh_p){
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 2, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    int idx = 0;
    for (const lf::mesh::Entity *corner : entity->SubEntities(2)) {
      masses(*corner) += area * (1.0/3.0);
//      if(idx == 0){
//        masses(*corner) += area * 26.0/1260.0;
//      }else if(idx == 1){
//        masses(*corner) += area * 1.0/60.0;
//      }else{
//        masses(*corner) += area * 1.0/60.0;
//      }
//      idx++;
    }
  }
//  for (const lf::mesh::Entity *entity : mesh_p->Entities(1)) {
//    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
//    double area = lf::geometry::Volume(*geo_ptr);
//    for (const lf::mesh::Entity *corner : entity->SubEntities(1)) {
//      masses(*corner) += area / 6.0;
//    }
//  }
  return masses;
}
/**
 * @brief Computes the masses m(p) of all midpoints of edges of the mesh
 * @param mesh_p pointer to the mesh
 * @return Datastructure containing the masses m(p) for all midpoints of edges of the mesh represented by mesh_p.
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMassesQuadraticEdges(std::shared_ptr<const lf::mesh::Mesh> mesh_p){
  lf::mesh::utils::CodimMeshDataSet<double> masses(mesh_p, 1, 0.0);
  // compute masses using a cell-based approach.
  for (const lf::mesh::Entity *entity : mesh_p->Entities(0)) {
    const lf::geometry::Geometry *geo_ptr = entity->Geometry();
    double area = lf::geometry::Volume(*geo_ptr);
    int idx = 0;
    for (const lf::mesh::Entity *corner : entity->SubEntities(1)) {
      masses(*corner) += area * 1.0/3.0;
//      if(idx == 0){
//        masses(*corner) += area * 20.0/63.0;
//      }else if(idx == 1){
//        masses(*corner) += area * 14.0/45.0;
//      }else{
//        masses(*corner) += area * 20.0/63.0;
//      }
//      idx++;
    }
  }
  return masses;
}
/**
 * @brief Enforce Dirichlet boundary conditions for quadratic FE space
 * @param fe_space
 * @param A
 * @param b
 */
void enforce_boundary_conditions(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fe_space,
    lf::assemble::COOMatrix<double>& A, Eigen::VectorXd& b){
  lf::mesh::utils::AllCodimMeshDataSet<bool> bd_flags(fe_space->Mesh(), false);
  // Loop over all edges
  for(const auto& edge : fe_space->Mesh()->Entities(1)){
    LF_ASSERT_MSG(edge->RefEl()==lf::base::RefEl::kSegment(),"Wrong codimension");
    const lf::geometry::Geometry* geo_ptr = edge->Geometry();
    const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
    // Check if edge is on the boundary
    if(corners(0,0) < 1e-5  ||
    corners(0,0) > 1. - 1e-5 ||
    corners(0,1) < 1e-5 ||
    corners(0,1) > 1. - 1e-5){
      bd_flags(*edge) = true;
    }
  }
  // Loop over all points
  for(const auto& point : fe_space->Mesh()->Entities(2)){
    LF_ASSERT_MSG(point->RefEl()==lf::base::RefEl::kPoint(),"Wrong codimension");
    const lf::geometry::Geometry* geo_ptr = point->Geometry();
    const Eigen::MatrixXd coords = lf::geometry::Corners(*geo_ptr);
    // Check if point is on the boundary
    if(coords(0) < 1e-5  ||
       coords(0) > 1. - 1e-5 ||
       coords(1) < 1e-5 ||
       coords(1) > 1. - 1e-5){
      bd_flags(*point) = true;
    }
  }
  // coefficient functions:
  // Dirichlet functor
  const auto g = [](const Eigen::Vector2d &x) {
    return x(1) == 0 ? 0.5 - std::abs(x(0) - 0.5) : 0.0;
  };
  lf::mesh::utils::MeshFunctionGlobal mf_g{g};
  auto flag_values{lf::fe::InitEssentialConditionFromFunction(*fe_space, bd_flags, mf_g)};

  lf::assemble::FixFlaggedSolutionCompAlt<double>(
      [&flag_values](lf::assemble::glb_idx_t dof_idx){
        return flag_values[dof_idx];
      }, A, b);
}
/**
 * @brief Computes the outward normals of a triangular cell
 * @param entity
 * @return Outward normals of the triangular cell
 */
Eigen::Matrix<double, 2, 3> computeOutwardNormalsTria(const lf::mesh::Entity &entity){
    Eigen::Matrix<double, 2, 3> normals;
    const lf::geometry::Geometry *geo_ptr = entity.Geometry();
    const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
//    Eigen::Matrix<double, 2, 3> midpoints;
//    midpoints << (corners.col(0) + corners.col(1))/2.0,
//                 (corners.col(1) + corners.col(2))/2.0,
//                 (corners.col(2) + corners.col(0))/2.0;

    Eigen::Matrix<double, 3, 3> delta;
    delta.block<3,2>(0,0) = corners.transpose();
    delta.block<3,1>(0,2) = Eigen::Vector3d::Ones();
    const auto det = delta.determinant();
    //rotation matrix of pi/2, 0, -1, 1, 0
    const Eigen::Matrix<double,2,2> rotation_matrix = (Eigen::Matrix<double,2,2>() << 0, -1, 1, 0).finished();
    for(int i = 0; i < 3; ++i){
        const Eigen::Vector2d edge = corners.col((i+1)%3) - corners.col(i);
        normals.col(i) = rotation_matrix * edge;
        normals.col(i) /= normals.col(i).norm();
    }
    if(det > 0){
      normals *= -1;
    }

    return normals;
}

/**
 * @brief Go through all cells and mark every upwind triangle with its corresponding interpolation point for which it is upwind
 * @param mesh_p Underlying mesh
 * @return CodimMeshDataset containing marked triangle global dofs with their corresponding interpolation point
 */
//lf::mesh::utils::CodimMeshDataSet<std::pair<bool,lf::assemble::gdof_idx_t>> markUpwinds(std::shared_ptr<const lf::mesh::Mesh> mesh_p){
//  lf::mesh::utils::CodimMeshDataSet<std::pair<bool,lf::assemble::gdof_idx_t>> upwind_triangles
//}

template <typename TMPMATRIX, class ENTITY_MATRIX_PROVIDER>
void AssembleMatrixGlobally(lf::assemble::dim_t codim, const lf::assemble::DofHandler &dofh, ENTITY_MATRIX_PROVIDER &entity_matrix_provider, TMPMATRIX &tmpmatrix){
  auto mesh = dofh.Mesh();

  lf::mesh::utils::CodimMeshDataSet<bool> upwind_triangles(mesh, 0, false);

  for(const lf::mesh::Entity *entity : mesh->Entities(codim)){
    if(entity_matrix_provider.isActive(*entity)){
      const lf::assemble::size_type num_local_dofs = dofh.NumLocalDofs(*entity);
      nonstd::span<const lf::assemble::gdof_idx_t> global_dof_idx = dofh.GlobalDofIndices(*entity);

      //make matrixXd of size num_local dofs
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> local_matrix(num_local_dofs, num_local_dofs);




      for(auto it = global_dof_idx.begin(); it != global_dof_idx.end(); ++it){
        for(auto jt = global_dof_idx.begin(); jt != global_dof_idx.end(); ++jt){



        }
      }
    }// end if(isActive)
  } //end main assembly loop
} //end AssembleMatrixGlobally

} // namespace UpwindQuadratureQuadratic