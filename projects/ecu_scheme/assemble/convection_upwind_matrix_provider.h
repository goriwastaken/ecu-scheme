//
// 
//

#ifndef THESIS_ASSEMBLE_CONVECTION_UPWIND_MATRIX_PROVIDER_H_
#define THESIS_ASSEMBLE_CONVECTION_UPWIND_MATRIX_PROVIDER_H_

#include "lf/fe/scalar_fe_space.h"
#include "lf/mesh/utils/utils.h"
#include "lf/uscalfe/uscalfe.h"
#include <Eigen/Core>

namespace ecu_scheme::assemble {

/**
 * @brief Computes the masses m(p) of all vertices of the mesh for the linear finite element space
 * @param mesh_p pointer to the mesh
 * @return data structure containing the masses m(p) for all vertices of the mesh represented by mesh_p
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMasses(const std::shared_ptr<const lf::mesh::Mesh>& mesh_p);

/**
 * @brief Computes the masses m(p) of all vertices of the mesh for the quadratic finite element space
 * @param mesh_p pointer to the mesh
 * @return data structure containing the masses m(p) for all vertices of the mesh represented by mesh_p
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMassesQuadratic(const std::shared_ptr<const lf::mesh::Mesh>& mesh_p);

/**
 * @brief Computes the masses m(p) of all edge midpoints of the mesh for the quadratic finite element space
 * @param mesh_p pointer to the mesh
 * @return data structure containing the masses m(p) for all edge midpoints of the mesh represented by mesh_p
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMassesQuadraticEdges(const std::shared_ptr<const lf::mesh::Mesh>& mesh_p);

/**
 * @brief Computes the outward normals of a triangular cell
 * @param entity Cell for which the outward normals should be computed
 * @return Matrix containing the outward normals of the cell
 */
Eigen::Matrix<double, 2, 3> computeOutwardNormalsTria(const lf::mesh::Entity &entity);

template <typename SCALAR, typename FUNCTOR>
class ConvectionUpwindMatrixProvider {
 public:
  /**
   * @brief Constructor for the ConvectionUpwindMatrixProvider class for the linear finite element space
   * @param fe_space reference to the finite element space
   * @param v functor for the velocity field
   */
  ConvectionUpwindMatrixProvider(std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space, FUNCTOR v, lf::mesh::utils::CodimMeshDataSet<double> masses);

  /**
   * @brief Constructor for the ConvectionUpwindMatrixProvider class for the quadratic finite element space
   * @param fe_space
   * @param v
   * @param masses_vertices
   * @param masses_edges
   */
  ConvectionUpwindMatrixProvider(std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space, FUNCTOR v, lf::mesh::utils::CodimMeshDataSet<double> masses_vertices, lf::mesh::utils::CodimMeshDataSet<double> masses_edges);

  /**
   * @brief main routine for the computation of element matrices of variable size depending on the finite element space
   * @param entity reference to the TRIANGULAR cell for which the element matrix is to be computed
   * @return an nxn matrix where n is the number of shape functions of the cell
   */
  Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> Eval(const lf::mesh::Entity &entity);

  /** @brief Default implementation: all cells are active */
  [[nodiscard]] bool isActive(const lf::mesh::Entity &/*entity*/) const { return true; }

 private:
  FUNCTOR v_; // velocity field
  std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space_; // finite element space
  lf::mesh::utils::CodimMeshDataSet<double> masses_vertices_; // masses of all vertex nodes of the mesh
  lf::mesh::utils::CodimMeshDataSet<double> masses_edges_; // masses of all edge midpoints of the mesh
};

template <typename SCALAR, typename FUNCTOR>
ConvectionUpwindMatrixProvider<SCALAR, FUNCTOR>::ConvectionUpwindMatrixProvider(
    std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space, FUNCTOR v, lf::mesh::utils::CodimMeshDataSet<double> masses)
    : fe_space_(std::move(fe_space)), v_(std::move(v)), masses_vertices_(std::move(masses)) {}

template <typename SCALAR, typename FUNCTOR>
ConvectionUpwindMatrixProvider<SCALAR, FUNCTOR>::ConvectionUpwindMatrixProvider(std::shared_ptr<const lf::fe::ScalarFESpace<
    SCALAR>> fe_space,
    FUNCTOR v,
    lf::mesh::utils::CodimMeshDataSet<double> masses_vertices,
    lf::mesh::utils::CodimMeshDataSet<double> masses_edges)
    : fe_space_(std::move(fe_space)), v_(std::move(v)), masses_vertices_(std::move(masses_vertices)), masses_edges_(std::move(masses_edges)) {}

template <typename SCALAR, typename FUNCTOR>
Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> ConvectionUpwindMatrixProvider<SCALAR,FUNCTOR>::Eval(const lf::mesh::Entity &entity) {
  LF_ASSERT_MSG(entity.RefEl() == lf::base::RefEl::kTria(),
                "Function only defined for triangular cells");
  const lf::geometry::Geometry *geo_ptr = entity.Geometry();
  const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
  const double area = lf::geometry::Volume(*geo_ptr);

  const size_t num_local_dofs = fe_space_->LocGlobMap().NumLocalDofs(entity);
  LF_ASSERT_MSG(num_local_dofs == 3 || num_local_dofs == 6, "Class supports only linear and quadratic FE spaces");
  //std::cout << "num local dofs: " << num_local_dofs << '\n';
  Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> element_matrix(num_local_dofs, num_local_dofs);

  // Compute normals of vertices of the triangular cell
  Eigen::Matrix<double, 3, 3> X;
  X.block<3,1>(0,0) = Eigen::Vector3d::Ones();
  X.block<3,2>(0,1) = corners.transpose();
  Eigen::Matrix<double, 2, 3> vertex_normals = -X.inverse().block<2,3>(1,0);

  // Compute edge midpoint coordinates - only needed for quadratic FE space
  Eigen::MatrixXd midpoints(2, 3);
  for(int i = 0; i < 3; ++i){
    midpoints.col(i) = 0.5 * (corners.col((i+1)%3) + corners.col(i));
  }

  // Prepare matrices for the computation of the element matrix and separate implementation for linear and quadratic FE space
  Eigen::MatrixXd velocities(2, num_local_dofs);
  Eigen::MatrixXd all_nodes(2, num_local_dofs);
  std::vector<double> local_masses;

  // Start case distinction linear or quadratic FE space
  if(num_local_dofs == 3) {
    // Linear FE space case
    all_nodes << corners;
    velocities << v_(all_nodes.col(0)), v_(all_nodes.col(1)), v_(all_nodes.col(2));
    for(const lf::mesh::Entity *e : entity.SubEntities(2)){
      local_masses.push_back(masses_vertices_(*e));
    }
    // the gradients in the linear case are obtained as the vertex normals, just with opposite sign
    Eigen::MatrixXd gradients_linearFE = -vertex_normals;

    for(int l=0; l < 3; ++l){
      Eigen::Vector2d vl = velocities.col(l);
      // Vertex a^l is upwind iff product of v(a^l) with both adjacent normals is positive
      if(vl.dot(vertex_normals.col((l+2)%3)) >= 0 && vl.dot(vertex_normals.col((l+1)%3)) >= 0){
        // a^l is upwind
        element_matrix.row(l) = local_masses[l] * vl.transpose() * gradients_linearFE;
      }else{
        // a^l is not upwind
        element_matrix.row(l) = Eigen::Vector<SCALAR, 3>::Zero();
      }
    }
    // End of linear FE space case
  }else{
    // Quadratic FE space
    all_nodes << corners, midpoints;
    velocities << v_(all_nodes.col(0)), v_(all_nodes.col(1)), v_(all_nodes.col(2)),
                  v_(all_nodes.col(3)), v_(all_nodes.col(4)), v_(all_nodes.col(5));

    for(const lf::mesh::Entity *e : entity.SubEntities(2)){
      local_masses.push_back(masses_vertices_(*e));
    }
    for(const lf::mesh::Entity *e : entity.SubEntities(1)){
      local_masses.push_back(masses_edges_(*e));
    }
    //std::cout << "local masses size: " << local_masses.size() << std::endl;
    //std::cout << "local masses: " << local_masses[0] << ", " << local_masses[1] << ", " << local_masses[2] << ", " << local_masses[3] << ", " << local_masses[4] << ", " << local_masses[5] << std::endl;
    LF_ASSERT_MSG(local_masses.size() == 6, "There must be six masses, one for each basis function");

    // Row-vector of barycentric coordinate functions based on global coordinates of nodes
    auto bary_functions = [area, corners](const Eigen::Vector2d& xh) -> Eigen::Matrix<double, 1, 3>{
      Eigen::Matrix<double, 1, 3> bary;
      const double coeff = 1.0 / (2.0 * area);
      // barycentric function 1
      bary.col(0) = coeff * (xh - corners.col(1)).transpose() * (Eigen::Vector2d(2,1) << corners(1,1) - corners(1,2), corners(0,2)-corners(0,1)).finished();
      // bary 2
      bary.col(1) = coeff * (xh - corners.col(2)).transpose() * (Eigen::Vector2d(2,1) << corners(1,2) - corners(1,0), corners(0,0)-corners(0,2)).finished();
      // bary 3
      bary.col(2) = coeff * (xh - corners.col(0)).transpose() * (Eigen::Vector2d(2,1) << corners(1,0) - corners(1,1), corners(0,1)-corners(0,0)).finished();
      return bary;
    };

    // Matrix of gradients of barycentric coordinate functions based on global coordinates of nodes
    auto bary_functions_grad = [area, corners](const Eigen::Vector2d& xh) -> Eigen::Matrix<double, 2, 3>{
      Eigen::Matrix<double, 2, 3> bary_grad;
      const double coeff = 1.0 / (2.0 * area);
      // bary 1
      bary_grad.col(0) = coeff * (Eigen::Vector2d(2,1) << corners(1,1) - corners(1,2), corners(0,2)-corners(0,1)).finished();
      // bary 2
      bary_grad.col(1) = coeff * (Eigen::Vector2d(2,1) << corners(1,2) - corners(1,0), corners(0,0)-corners(0,2)).finished();
      // bary 3
      bary_grad.col(2) = coeff * (Eigen::Vector2d(2,1) << corners(1,0) - corners(1,1), corners(0,1)-corners(0,0)).finished();
      return bary_grad;
    };

    // Matrix of gradients of local shape functions based on global coordinates of nodes in quadratic Lagrangian FE space
    auto gradientsLocalShapeFunctions = [bary_functions, bary_functions_grad]
        (const Eigen::Vector2d xh) -> Eigen::Matrix<double, 2, 6>{
      Eigen::Matrix<double, 2, 6> gradients;
      // barycentric coordinate functions
      Eigen::Matrix<double, 1, 3> temp = bary_functions(xh);
      Eigen::Matrix<double, 2, 3> grads_bary = bary_functions_grad(xh);
      Eigen::RowVector3d l;
      l << temp(0,0), temp(0,1), temp(0,2);

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

    // Main loop for the computation of the element matrix
    for(int l=0; l < 6; ++l){
      Eigen::Vector2d vl = velocities.col(l);
      // compute product of v(a^l) with gradients of basis function of a^l
      Eigen::Matrix<double, 2, 6> grads = gradientsLocalShapeFunctions(all_nodes.col(l));
      Eigen::Matrix<double, 1, 6> contribution = vl.transpose() * grads;

      // Vertex a^l is upwind iff product of v(a^l) with both adjacent normals is positive
      if(l < 3){
        // first 3 nodes are vertices of triangle
        if(vl.dot(vertex_normals.col((l+2)%3)) >= 0 && vl.dot(vertex_normals.col((l+1)%3)) >= 0){
          // a^l is upwind
          element_matrix.row(l) = local_masses[l] * contribution;
        }else{
            // a^l is not upwind
            element_matrix.row(l) = Eigen::Vector<SCALAR, 6>::Zero();
        }
      }else{
        // last 3 nodes are edge midpoints of triangle
        const Eigen::Matrix<double, 2, 3> outward_normals = computeOutwardNormalsTria(entity);
        // Midpoint m^l is upwind iff product of v(m^l) with corresponding outward normals is positive
        if(vl.dot(outward_normals.col(l%3)) >= 0){
          // m^l is upwind
          element_matrix.row(l) = local_masses[l] * contribution;
        }else{
          // m^l is not upwind
          element_matrix.row(l) = Eigen::Vector<SCALAR, 6>::Zero();
        }
      }
    }
    // End of quadratic FE space case
  } // End case distinction linear or quadratic FE space

  return element_matrix;
}


} // assemble

#endif //THESIS_ASSEMBLE_CONVECTION_UPWIND_MATRIX_PROVIDER_H_
