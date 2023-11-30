// Project header file include
// C system headers
// C++ standard library headers
// Other libraries headers
#ifndef THESIS_ASSEMBLE_SUPG_MATRIX_PROVIDER_H_
#define THESIS_ASSEMBLE_SUPG_MATRIX_PROVIDER_H_

#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"


namespace ecu_scheme::assemble {

// todo implement SUPG matrix provider based on homework problems
template<class MESHFUNCTION_V>
class SUPGElementMatrixProvider{
 public:
  SUPGElementMatrixProvider(const SUPGElementMatrixProvider &) = delete;
  SUPGElementMatrixProvider(SUPGElementMatrixProvider &&) noexcept = default;
  SUPGElementMatrixProvider &operator=(const SUPGElementMatrixProvider &) = delete;
  SUPGElementMatrixProvider &operator=(SUPGElementMatrixProvider &&) = delete;

  explicit SUPGElementMatrixProvider(MESHFUNCTION_V &velocity, bool use_delta = true);

  virtual bool isActive(const lf::mesh::Entity & /*cell*/) { return true; }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Eval(const lf::mesh::Entity &cell);

  virtual ~SUPGElementMatrixProvider() = default;
 private:
  const lf::quad::QuadRule qr_{lf::quad::make_TriaQR_P6O4()};
  // mesh function providing the velocity field
  MESHFUNCTION_V &velocity_;
  // Values of reference shape functions at quadrature points
  Eigen::MatrixXd val_ref_lsf_;
  // Gradients of reference shape functions at quadrature points
  Eigen::MatrixXd grad_ref_lsf_;
  // Flag for controlling use of delta scaling
  bool use_delta_;
};

template<class MESHFUNCTION_V>
SUPGElementMatrixProvider<MESHFUNCTION_V>::SUPGElementMatrixProvider(MESHFUNCTION_V &velocity, bool use_delta)
    : velocity_(velocity), use_delta_(use_delta) {
  const lf::uscalfe::FeLagrangeO2Tria<double> ref_fe_space;
  LF_ASSERT_MSG(ref_fe_space.RefEl() == lf::base::RefEl::kTria(),
                "This class only works for triangular elements");
  LF_ASSERT_MSG(ref_fe_space.NumRefShapeFunctions() == 6,
                "Quadratic Lagrange FE must have 6 local shape functions");
  LF_ASSERT_MSG(qr_.RefEl() == lf::base::RefEl::kTria(),
                "Quadrature rule only for triangular cells");
  LF_ASSERT_MSG(qr_.NumPoints() == 6, "Quadrature rule must have 6 points");
  // Obtain values and gradients of reference shape functions at
  // quadrature nodes, see \lref{par:lfppparfe}
  val_ref_lsf_ = ref_fe_space.EvalReferenceShapeFunctions(qr_.Points());
  grad_ref_lsf_ = ref_fe_space.GradientsReferenceShapeFunctions(qr_.Points());
}

template<class MESHFUNCTION_V>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> SUPGElementMatrixProvider<MESHFUNCTION_V>::Eval(const lf::mesh::Entity &cell) {
  // For quadratic Lagrange FE we have exactly 6 local shape functions
  // Element matrix is a 6x6 matrix
  Eigen::Matrix<double, 6, 6> elem_mat;
  elem_mat.setZero();

  LF_ASSERT_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Only implemented for triangles");
  // Obtain geometry information
  const lf::geometry::Geometry *geo_ptr = cell.Geometry();
  LF_ASSERT_MSG(geo_ptr->DimGlobal() == 2,
                "Only implemented for planar triangles");
  // Gram determinant at quadrature points
  const Eigen::VectorXd dets(geo_ptr->IntegrationElement(qr_.Points()));
  LF_ASSERT_MSG(dets.size() == qr_.NumPoints(),
                "Mismatch " << dets.size() << " <-> " << qr_.NumPoints());
  // Fetch the transformation matrices for the gradients
  const Eigen::MatrixXd JinvT(geo_ptr->JacobianInverseGramian(qr_.Points()));
  LF_ASSERT_MSG(JinvT.cols() == 2 * qr_.NumPoints(),
                "Mismatch " << JinvT.cols() << " <-> " << 2 * qr_.NumPoints());
  // Obtain values of velocity at the 6 quadrature points
  std::vector<Eigen::Vector2d> v_vals = velocity_(cell, qr_.Points());

  //todo two ways to compute delta coefficient, figure out if delta or delta^(1/2) is better
  const Eigen::MatrixXd vertices{lf::geometry::Corners(*geo_ptr)};
  // size of triangle
  const double hK = std::max({(vertices.col(0) - vertices.col(1)).norm(),
                              (vertices.col(1) - vertices.col(2)).norm(),
                              (vertices.col(2) - vertices.col(0)).norm()});
  // max modulus of velocity in quad nodes
  const double velocity_max = std::max_element(v_vals.begin(), v_vals.end(),
                                               [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                                                 return a.norm() < b.norm();
                                               })->norm();
  const double kDelta = use_delta_ ? hK / (2.0 * velocity_max) : 1.0;
  // End compute the delta coefficient

  for(int l = 0; l < 6; ++l){
    // Metric factor $\cob{|\det\Derv\Phibf_K(\wh{\zetabf}_{\ell})}$ scaled with
    // quadrature weight $\cob{\omega_{\ell}}$
    const double kFactor = qr_.Weights()[l] * dets[l];
    // Compute transformed gradients $\cob{\Vt_{\ell}^i}$ of all local shape
    // functions and collect them in the columns of a 2 x 6-matrix
    const Eigen::Matrix<double, 2, 6> kTrfGrad = JinvT.block(0, 2 * l, 2, 2) * (grad_ref_lsf_.block(0, 2 * l, 6, 2).transpose());
    // Compute the inner products of the transformed gradients with the
    // velocity vector at the current quadrature point.
    const Eigen::Matrix<double, 1, 6> mvec = v_vals[l].transpose() * kTrfGrad;
    //todo check if it's worth splitting into diffusive part and convective part
    // loop and assemble the element matrix without relying on DiffusiveElementMatrixProvider for the diffusive part
    for(int j = 0; j < 6; ++j){
      for(int i = 0; i < 6; ++i){
        elem_mat(i, j) +=
            kFactor * (kDelta * mvec[i] * mvec[j] + mvec[j] * val_ref_lsf_(i, l));
      }
    }
  }
  return elem_mat;
}

} // assemble

#endif //THESIS_ASSEMBLE_SUPG_MATRIX_PROVIDER_H_
