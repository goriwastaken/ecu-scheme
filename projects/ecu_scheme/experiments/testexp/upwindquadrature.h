#ifndef UPWIND_QUADRATURE_H
#define UPWIND_QUADRATURE_H

/**
 * @file upwindquadrature.h
 * @brief NPDE homework template
 * @author Philippe Peter
 * @date June 2020
 * @copyright Developed at SAM, ETH Zurich
 */
#include <lf/base/base.h>
#include <lf/geometry/geometry.h>
#include <lf/mesh/mesh.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <memory>
#include <vector>

namespace UpwindQuadrature {

/**
 * @brief Computes the masses m(p) of all vertices of the mesh
 * @param mesh_p pointer to the mesh.
 * @return data structure containing the masses m(p) for all vertices p of the
 * mesh represented by mesh_p.
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMasses(
    std::shared_ptr<const lf::mesh::Mesh> mesh_p);

/**
 * @headerfile upwindquadrature.h
 * @brief Computes the local matrices for the convection term based on the
 * upwind quadrature scheme introced in the exercise.
 * @tparam FUNCTOR function that defines the vector valued velocity
 * coefficient v.
 */
template <typename FUNCTOR>
class UpwindConvectionElementMatrixProvider {
 public:
  /**
   * @brief
   * @param v functor for the velocity field
   * @param masses data structure storing the masses m(a^j) for all vertices of
   * the mesh.
   */
  explicit UpwindConvectionElementMatrixProvider(
      FUNCTOR v, lf::mesh::utils::CodimMeshDataSet<double> masses)
      : v_(v), masses_(masses) {}

  /**
   * @brief main routine for the computation of element matrices.
   * @param entity reference to the TRIANGULAR cell for which the element
   * matrix should be computed.
   * @return a 3x3 matrix containing the element matrix.
   */
  Eigen::Matrix3d Eval(const lf::mesh::Entity &entity);

  /** @brief Default implementation: all cells are active */
  bool isActive(const lf::mesh::Entity & /*entity*/) const { return true; }

 private:
  FUNCTOR v_;  // velocity field
  lf::mesh::utils::CodimMeshDataSet<double>
      masses_;  // masses of all vertices of the mesh.
};

/* SAM_LISTING_BEGIN_1 */
template <typename FUNCTOR>
Eigen::Matrix3d UpwindConvectionElementMatrixProvider<FUNCTOR>::Eval(
    const lf::mesh::Entity &entity) {
  LF_ASSERT_MSG(lf::base::RefEl::kTria() == entity.RefEl(),
                "Function only defined for triangular cells");

  const lf::geometry::Geometry *geo_ptr = entity.Geometry();
  const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
  const double area = lf::geometry::Volume(*geo_ptr);
  Eigen::Matrix3d loc_mat;

  //====================
  Eigen::Matrix<double, 3, 3> X;
  X.block<3,1>(0,0) = Eigen::Vector3d::Ones();
  X.block<3,2>(0,1) = corners.transpose();
  Eigen::Matrix<double, 2, 3> grads = X.inverse().block<2,3>(1,0);

  std::vector<double> local_masses;
  for(const lf::mesh::Entity *e : entity.SubEntities(2)){
    local_masses.push_back(masses_(*e));
  }

  Eigen::MatrixXd velocities(2,3);
  velocities << v_(corners.col(0)), v_(corners.col(1)), v_(corners.col(2));

  Eigen::MatrixXd n = -grads; //not normalized

  //-v(a^j) is upwind iff product with both adjacent normals is positive
  //check first node
  Eigen::Vector2d v0 = velocities.col(0);
  if(v0.dot(n.col(1)) > 0 && v0.dot(n.col(2)) > 0){

      loc_mat.row(0) = local_masses[0] * velocities.transpose().row(0)*grads;
      LF_ASSERT_MSG(loc_mat.row(0).sum() < 1e-10, "Row sum must be zero");

  }else{
    loc_mat.row(0) = Eigen::Vector3d::Zero();
  }
  Eigen::Vector2d v1 = velocities.col(1);
  if(v1.dot(n.col(0)) > 0 && v1.dot(n.col(2)) > 0){

      loc_mat.row(1) = local_masses[1] * velocities.transpose().row(1)*grads;
      LF_ASSERT_MSG(loc_mat.row(1).sum() < 1e-10, "Row sum must be zero");

  }else{
    loc_mat.row(1) = Eigen::Vector3d::Zero();
  }
  Eigen::Vector2d v2 = velocities.col(2);
  if(v2.dot(n.col(1)) > 0 && v2.dot(n.col(0)) > 0){

      loc_mat.row(2) = local_masses[2] * velocities.transpose().row(2)*grads;
      LF_ASSERT_MSG(loc_mat.row(2).sum() < 1e-10, "Row sum must be zero");

  }else{
    loc_mat.row(2) = Eigen::Vector3d::Zero();
  }

  //====================
  return loc_mat;
}
/* SAM_LISTING_END_1 */

}  // namespace UpwindQuadrature

namespace UpwindQuadratureQuadratic {
/**
 * @brief Computes the masses m(p) of all vertices of the mesh
 * @param mesh_p pointer to the mesh
 * @return data structure containinng the masses m(p) for all interpolation points p of the mesh represented by mesh_p
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMassesQuadratic(
    std::shared_ptr<const lf::mesh::Mesh> mesh_p);

/**
 * @brief Computes the masses m(p) of all midpoints of edges of the mesh
 * @param mesh_p pointer to the mesh
 * @return Data structure containing the masses m(p) for all midpoints of edges of the mesh represented by mesh_p
 */
lf::mesh::utils::CodimMeshDataSet<double> initializeMassesQuadraticEdges(
    std::shared_ptr<const lf::mesh::Mesh> mesh_p);

/**
 * @brief Enforces the boundary conditions for the upwind quadrature scheme
 * @param fe_space quadratic FE space
 * @param A Galerkin matrix for the convective term
 * @param b RHS vector
 */
void enforce_boundary_conditions(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fe_space,
    lf::assemble::COOMatrix<double>& A, Eigen::VectorXd& b);
/**
 * @brief Computes the outward normals of a triangular cell
 * @param entity Entity for which the outward normals should be computed
 * @return Matrix containing the outward normals of the vertices of the cell in its columns
 */
Eigen::Matrix<double, 2, 3> computeOutwardNormalsTria(const lf::mesh::Entity &entity);

/**
 * @headerfile upwindquadrature.h
 * @brief Computes the local matrices for the convection term based on the upwind quadrature scheme in quadratic Lagrangian FE Space
 * @tparam FUNCTOR function that defines the vector valued velocity coefficient v
 */
template <typename FUNCTOR>
class UpwindQuadratureQuadraticElementMatrixProvider{
 public:
  /**
   * @brief Constructor
   * @param v functor for the velocity field
   * @param masses data structure for storing the masses m(a^j) for every interpolation node of the mesh
   */
  explicit UpwindQuadratureQuadraticElementMatrixProvider(
      FUNCTOR v, lf::mesh::utils::CodimMeshDataSet<double> masses, lf::mesh::utils::CodimMeshDataSet<double> masses_edges)
      : v_(v), masses_(masses), masses_edges_(masses_edges) {
  }

  /**
   * @brief main routine for the computation of element matrices.
   * @param entity reference to the TRIANGULAR cell for which the element
   * matrix should be computed.
   * @return a 6x6 matrix containing the element matrix.
   */
  Eigen::Matrix<double, 6, 6> Eval(const lf::mesh::Entity &entity);

  /** @brief Default implementation: all cells are active */
  bool isActive(const lf::mesh::Entity & /*entity*/) const { return true; }

 private:
  FUNCTOR v_; // velocity field
  lf::mesh::utils::CodimMeshDataSet<double>
      masses_; // masses of all vertex points of the mesh
  lf::mesh::utils::CodimMeshDataSet<double>
      masses_edges_; // masses of all midpoints of edges of the mesh
};
template<typename FUNCTOR>
Eigen::Matrix<double, 6, 6> UpwindQuadratureQuadraticElementMatrixProvider<FUNCTOR>::Eval(const lf::mesh::Entity &entity) {
  LF_ASSERT_MSG(lf::base::RefEl::kTria() == entity.RefEl(),
                "Function only defined for triangular cells");

  const lf::geometry::Geometry *geo_ptr = entity.Geometry();
  const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
  const double area = lf::geometry::Volume(*geo_ptr);
  Eigen::Matrix<double, 6, 6> loc_mat = Eigen::Matrix<double, 6, 6>::Zero();
  //std::cout << corners.size() << '\n';
  // compute midpoints of edges
  Eigen::MatrixXd midpoints(2,3);
  for(int i=0;i<3;++i){
    midpoints.col(i) = 0.5 * (corners.col(i) + corners.col((i+1)%3));
  }

  // Matrix containing all vertex and edge midpoint coordinates
  Eigen::MatrixXd all_points(2,6);
  all_points << corners, midpoints;

  auto bary_functions = [area, corners](const Eigen::Vector2d xh) -> Eigen::Matrix<double, 1, 3> {
    Eigen::Matrix<double, 1, 3> bary;
    const double coeff = 1.0 / (2.0 * area);
    // bary 1
    //std::cout << (xh - corners.col(1)).transpose().rows() << ' ' << (xh - corners.col(1)).transpose().cols() << "-------------" << '\n';
    //auto dbg_mat = (Eigen::Vector2d(2,1) << corners(1,1) - corners(2,1), corners(2,0)-corners(1,0)).finished();
    //std::cout << dbg_mat.rows() << dbg_mat.cols() << " ===================== " << '\n';
    bary.col(0) = coeff * (xh - corners.col(1)).transpose() * (Eigen::Vector2d(2,1) << corners(1,1) - corners(1,2), corners(0,2)-corners(0,1)).finished();
    // bary 2
    bary.col(1) = coeff * (xh - corners.col(2)).transpose() * (Eigen::Vector2d(2,1) << corners(1,2) - corners(1,0), corners(0,0)-corners(0,2)).finished();
    // bary 3
    bary.col(2) = coeff * (xh - corners.col(0)).transpose() * (Eigen::Vector2d(2,1) << corners(1,0) - corners(1,1), corners(0,1)-corners(0,0)).finished();
    return bary;
  };
  auto bary_functions_grad = [area, corners](const Eigen::Vector2d xh) -> Eigen::Matrix<double, 2, 3> {
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

  // Compute basis functions of quadratic Lagrangian FE space at given points
  Eigen::Matrix<double, 6, 6> bsf;
  for(int i=0;i<6;++i){
    //Eigen::Vector2d p = all_points.col(i);
    // Convert to array type for componentwise operations
    auto x0 = all_points.row(0).array();
    auto x1 = all_points.row(1).array();
    // Evaluation of basis function formulas
    bsf.row(0) = (2.0 * (1 - x0 - x1) * (0.5 - x0 - x1)).matrix();
    bsf.row(1) = (2.0 * x0 * (x0 - 0.5)).matrix();
    bsf.row(2) = (2.0 * x1 * (x1 - 0.5)).matrix();
    bsf.row(3) = (4.0 * (1 - x0 - x1) * x0).matrix();
    bsf.row(4) = (4.0 * x0 * x1).matrix();
    bsf.row(5) = (4.0 * (1 - x0 - x1) * x1).matrix();
  }
  // Compute gradients of basis functions of quadratic Lagrangian FE space at given points
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> grad_bsf(6, 12);
  //reshape into a 12x6 matrix
  Eigen::Map<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>, Eigen::AutoAlign> grad_bsf_map(grad_bsf.data(), 12, 6);
  for(int i=0;i<6;++i){
    //Eigen::Vector2d p = all_points.col(i);
    // Convert to array type for componentwise operations
    auto x0 = all_points.row(0).array();
    auto x1 = all_points.row(1).array();
    // Evaluation of basis function formulas
    // d/dx_0
    grad_bsf_map.row(0) = -3.0 + 4.0 * x0 + 4.0 * x1;
    grad_bsf_map.row(1) = 4.0 * x0 - 1.0;
    grad_bsf_map.row(2) = 0.0;
    grad_bsf_map.row(3) = 4.0 - 8.0 * x0 - 4.0 * x1;
    grad_bsf_map.row(4) = 4.0 * x1;
    grad_bsf_map.row(5) = -4.0 * x1;
    // d/dx_1
    grad_bsf_map.row(6) = -3.0+4.0*x0+4.0*x1;
    grad_bsf_map.row(7) = 0.0;
    grad_bsf_map.row(8) = 4.0*x1 - 1.0;
    grad_bsf_map.row(9) = -4.0 * x0;
    grad_bsf_map.row(10) = 4.0 * x0;
    grad_bsf_map.row(11) = 4.0 - 4.0 * x0 - 8.0 * x1;
  }

  // Normal vectors of triangle are accesible through barycentric coordinates of triangle
  Eigen::Matrix<double, 3, 3> X;
  X.block<3,1>(0,0) = Eigen::Vector3d::Ones();
  X.block<3,2>(0,1) = corners.transpose();
  Eigen::Matrix<double, 2, 3> grads_bary = X.inverse().block<2,3>(1,0);
  Eigen::Matrix<double, 2, 3> vertex_normals;
  vertex_normals = -grads_bary;


  // Compute velocities
  Eigen::MatrixXd velocities(2, 6);
  velocities << v_(corners.col(0)), v_(corners.col(1)), v_(corners.col(2)),
                v_(midpoints.col(0)), v_(midpoints.col(1)), v_(midpoints.col(2));

  // compute masses
  std::vector<double> local_masses;
  for(const lf::mesh::Entity *e : entity.SubEntities(2)) {
    local_masses.push_back(masses_(*e));
  }
  for(const lf::mesh::Entity *e : entity.SubEntities(1)){
    local_masses.push_back(masses_edges_(*e));
  }
  LF_ASSERT_MSG(local_masses.size() == 6, "There must be six masses, one for each node");
  // todo check masses

//  Eigen::Matrix<double, 6, 6> Y;
//  Y.block<6,1>(0,0) = Eigen::Vector<double,6>::Ones();
//  Y.block<6,2>(0,1) = all_points.transpose();
//  Eigen::Matrix<double, 6, 6> inverse;
//  std::cout << " Determinant is: " << Y.determinant() << std::endl;
//  Eigen::Matrix<double, 2, 6> grads_Y = Y.inverse().block<2,6>(1,0);
//  Eigen::Matrix<double, 2, 6> interp_normals;
//  interp_normals = -grads_Y;


  // -v(a^j) is upwind iff product with both adjacent normals is positive
//  for(int l=0;l<6;++l){
//    // loop over interpolation points
//    Eigen::Vector2d vl = velocities.col(l);
//    if(vl.dot(vertex_normals.col((l+2)%3)) >= 1e-5){
//      const Eigen::Matrix<double, 1, 6> mvec = vl.transpose()*grad_bsf.block(0, 2*l, 6, 2).transpose();
//      //const Eigen::Matrix<double, 1, 6> mvec = vl.transpose()*grads_Y;
//      //if midpoint we are done
//      if(l >= 3){
////        for(int j=0;j<6;++j){
////          for(int i=0;i<6;++i){
////            loc_mat(i, j) += local_masses[l] * mvec[j] * bsf(i, l);
////          }
////        }
//        //loc_mat.row(l) = local_masses[l]* bsf(l,l) * mvec ;
//        loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
////        LF_ASSERT_MSG(loc_mat.row(l).sum() < 1e-10, "Row sum must be zero");
//        //LF_ASSERT_MSG(loc_mat.col(l).sum() < 1e-10, "Col sum must be zero");
//      }
//      //if vertex we need second adjacent normal test
//      else if(l < 3 && vl.dot(vertex_normals.col((l+1)%3)) > 1e-5){
////        for(int j=0;j<6;++j){
////          for(int i=0;i<6;++i){
////            loc_mat(i, j) -= local_masses[l] * mvec[j] * bsf(i, l);
////          }
////        }
//        //loc_mat.row(l) = local_masses[l]* (-bsf(l,l)) * mvec;
//        Eigen::Matrix<double, 1, 3> temp = local_masses[l] * vl.transpose() * grads_Y.block(0, 0, 2, 3);
//        //augument temp with 3 zero columns
//        Eigen::Matrix<double, 1, 6> temp_aug;
//        temp_aug << temp, Eigen::Matrix<double, 1, 3>::Zero();
//        loc_mat.row(l) = temp_aug;
//       // LF_ASSERT_MSG(loc_mat.row(l).sum() < 1e-10, "Row sum must be zero");
//
//      }
//      else{
//        loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
//      }
//    }else{
//        loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
//    }
//  }


  auto gradientsLocalShapeFunctions =
      [/*&grads_bary,*/ bary_functions, bary_functions_grad](
          const Eigen::Vector2d xh) -> Eigen::Matrix<double, 2, 6> {
        Eigen::Matrix<double, 2, 6> gradients;
        // barycentric coordinate functions
//        const std::array<double, 3> l{1.0 - xh[0] - xh[1], xh[0], xh[1]};
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
  auto interp_normals =
      [grad_bsf](int idx) -> Eigen::Matrix<double, 2, 6>{
    return grad_bsf.block(0, 2*idx, 6, 2).transpose();
  };

  //    Eigen::Matrix<double, 2, 6> new_grads = gradientsLocalShapeFunctions(all_points.col(l));
//    Eigen::Matrix<double, 1, 6> mvec = vl.transpose()*new_grads;
// set first three entries of mvec to 0 into a newvec and last three take from mvec
//  Eigen::Matrix<double, 1, 6> newvec = Eigen::Matrix<double, 1, 6>::Zero();
//  newvec << 0., 0., 0., mvec(3), mvec(4), mvec(5);

  for(int l=0;l<6;++l){
    Eigen::Vector2d vl = velocities.col(l);
    // compute product of v(a^l) with gradients of local shape functions
    //const Eigen::Matrix<double, 1, 6> mvec = vl.transpose()*grad_bsf.block(0, 2*l, 6, 2).transpose();
    Eigen::Matrix<double, 2, 6> new_grads = gradientsLocalShapeFunctions(all_points.col(l));
    Eigen::Matrix<double, 1, 6> mvec = vl.transpose()*new_grads;
    //node is in upwind direction if dot product of v(a^l) with both adjacent normals is positive
    if(l<3){
      //vertices
      if(vl.dot(vertex_normals.col((l+2)%3)) >= 0  && vl.dot(vertex_normals.col((l+1)%3)) >= 0) {
        loc_mat.row(l) = local_masses[l] * mvec;
      }else{
        loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
      }
    }else{
      //midpoints
      const Eigen::Matrix<double, 2, 3> outwardnormals = computeOutwardNormalsTria(entity);
      // upwind if dot product of v(m^l) with adjacent outward normals of triangle is positive
      if(/*vl.dot(outwardnormals.col((l+1)%3)) >= 0 &&*/ vl.dot(outwardnormals.col((l)%3)) >= 0){
      //if(vl.dot(outwardnormals.col((l+2)%3)) >= 0 && vl.dot(outwardnormals.col((l)%3)) >= 0 ){
        loc_mat.row(l) = local_masses[l] * mvec;
      }else{
        loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
      }
    }
  }

// ============================================================================
//  //vertices
//  for(int l=0;l<3;++l){
//    Eigen::Vector2d vl = velocities.col(l);
//    //const Eigen::Matrix<double, 1, 6> mvec = vl.transpose()*grad_bsf.block(0, 2*l, 6, 2).transpose();
//    const Eigen::Matrix<double, 2, 6> new_grads = gradientsLocalShapeFunctions(all_points.col(l));
//    const Eigen::Matrix<double, 1, 6> tempmvec = vl.transpose()*new_grads;
//
//    Eigen::Matrix<double, 1, 6> mvec;
//    const Eigen::Matrix<double, 1, 3> temp = vl.transpose() * grads_bary;
//    for(int k=0;k<6;++k){
//      if(k < 3)
//        mvec.col(k) = temp.col(k);
//      else
//        mvec.col(k) = tempmvec.col(k) ;
//    }
//
//    if( vl.dot(vertex_normals.col((l+2)%3)) > 0  && vl.dot(vertex_normals.col((l+1)%3)) > 0){
//      loc_mat.row(l) = local_masses[l] *  mvec;
//    }else{
//      loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
//    }
//  }
//  // boolean upwind store if vertex is upwind iff product with both adjacent normals is positive
//  std::vector<bool> upwind_vertex(3);
//  for(int l=0;l<3;++l){
//    Eigen::Vector2d vl = velocities.col(l);
//    upwind_vertex[l] = vl.dot(vertex_normals.col((l+2)%3)) > 0  && vl.dot(vertex_normals.col((l+1)%3)) > 0;
//  }
//  // boolean upwind based on outwardnormals
//  std::vector<bool> upwind(3);
//  const Eigen::Matrix<double, 2, 3> outwardnormals = computeOutwardNormalsTria(entity);
//  for(int l=0;l<3;++l){
//    Eigen::Vector2d vl = velocities.col(l);
//    upwind[l] = vl.dot(outwardnormals.col(l)) > 0;
//  }
//
//  //midpoints
//  for(int l=3;l<6;++l){
//    Eigen::Vector2d vl = velocities.col(l);
//    const Eigen::Matrix<double, 1, 6> mvec = vl.transpose()*grad_bsf.block(0, 2*l, 6, 2).transpose();
//    //const Eigen::Matrix<double, 2, 6> new_grads = gradientsLocalShapeFunctions(all_points.col(l));
//    //const Eigen::Matrix<double, 1, 6> tempmvec = vl.transpose()*new_grads;
//    //vl.dot(vertex_normals.col(l%3)) > 0  && vl.dot(vertex_normals.col((l+1)%3)) > 0
//    //Eigen::Matrix<double, 1, 6> mvec;
////    for(int k=0;k<6;++k){
////      mvec.col(k) = tempmvec.col(k) * (bsf(l,k));
////    }
//    //if(upwind[l%3] && upwind[(l+1)%3]){
//    if( vl.dot(vertex_normals.col((l+2)%3)) > 0  && vl.dot(vertex_normals.col((l)%3)) > 0){
//      loc_mat.row(l) = local_masses[l] * mvec;
//    }else{
//      loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
//    }
//  }

  // ============================================================================

//  for(int l=3;l<6;++l){
//    Eigen::Vector2d vl = velocities.col(l);
//    //const Eigen::Matrix<double, 1, 6> mvec = vl.transpose()*grad_bsf.block(0, 2*l, 6, 2).transpose();
//    const Eigen::Matrix<double, 2, 6> new_grads = gradientsLocalShapeFunctions(all_points.col(l));
//    const Eigen::Matrix<double, 1, 6> tempmvec = vl.transpose()*new_grads;
//    //vl.dot(vertex_normals.col(l%3)) > 0  && vl.dot(vertex_normals.col((l+1)%3)) > 0
//    Eigen::Matrix<double, 1, 6> mvec;
//    for(int k=0;k<6;++k){
//      mvec.col(k) = tempmvec.col(k) * (bsf(l,k));
//    }
//    //if(upwind[l%3] && upwind[(l+1)%3]){
//    if( (upwind_vertex[l%3] && upwind_vertex[(l+1)%3])){
//      loc_mat.row(l) = local_masses[l] * mvec;
//    }else{
//      loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
//    }
//  }

//  for(int l=0;l<6;++l) {
//    Eigen::Vector2d vl = velocities.col(l);
//    if (vl.dot(vertex_normals.col((l+2)%3)) >= 1e-6) {
//      //const Eigen::Matrix<double, 1, 6> mvec = vl.transpose()*grad_bsf.block(0, 2*l, 6, 2).transpose();
//      const Eigen::Matrix<double, 1, 6> mvec = vl.transpose() * grads_Y;
//      //if midpoint we are done
//      if (l >= 3) {
//        loc_mat.row(l) = local_masses[l] *mvec;
//      }
//        //if vertex we need second adjacent normal test
//      else if (l < 3 && vl.dot(vertex_normals.col((l + 1) % 3)) >= 1e-6) {
//        loc_mat.row(l) = local_masses[l]  *mvec;
//      } else {
//        loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
//      }
//    }else{
//      loc_mat.row(l) = Eigen::Vector<double, 6>::Zero();
//    }
//  }

//  LF_ASSERT_MSG(loc_mat.col(0).sum() < 1e-10, "Col sum must be zero");
//  LF_ASSERT_MSG(loc_mat.col(1).sum() < 1e-10, "Col sum must be zero");
//  LF_ASSERT_MSG(loc_mat.col(2).sum() < 1e-10, "Col sum must be zero");

  return loc_mat;
}

} // namespace UpwindQuadratureQuadratic

#endif  // UPWIND_QUADRATURE_H
