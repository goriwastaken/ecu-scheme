// Project header file include
// C system headers
// C++ standard library headers
// Other libraries headers
#ifndef THESIS_ASSEMBLE_SUPG_MATRIX_PROVIDER_H_
#define THESIS_ASSEMBLE_SUPG_MATRIX_PROVIDER_H_

#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <memory>

#include "bdc_utils.h"
#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"

namespace ecu_scheme::assemble {

/**
 * @brief Assemble the SUPG element matrix for the convection-diffusion boundary
 * for quadratic Lagrange FE value problem where the implementation is based on
 * a combination of techniques from the NumPDE lecture document
 * @tparam MESHFUNCTION_V type of mesh function providing the velocity field
 */
template <class MESHFUNCTION_V>
class SUPGElementMatrixProvider {
 public:
  SUPGElementMatrixProvider(const SUPGElementMatrixProvider &) = delete;
  SUPGElementMatrixProvider(SUPGElementMatrixProvider &&) noexcept = default;
  SUPGElementMatrixProvider &operator=(const SUPGElementMatrixProvider &) =
      delete;
  SUPGElementMatrixProvider &operator=(SUPGElementMatrixProvider &&) = delete;

  explicit SUPGElementMatrixProvider(MESHFUNCTION_V &velocity,
                                     bool use_delta = true);

  virtual bool isActive(const lf::mesh::Entity & /*cell*/) { return true; }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Eval(
      const lf::mesh::Entity &cell);

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

template <class MESHFUNCTION_V>
SUPGElementMatrixProvider<MESHFUNCTION_V>::SUPGElementMatrixProvider(
    MESHFUNCTION_V &velocity, bool use_delta)
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

template <class MESHFUNCTION_V>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
SUPGElementMatrixProvider<MESHFUNCTION_V>::Eval(const lf::mesh::Entity &cell) {
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

  // two ways to compute delta coefficient, we use delta instead of delta^(1/2)
  const Eigen::MatrixXd vertices{lf::geometry::Corners(*geo_ptr)};
  // size of triangle
  const double hK = std::max({(vertices.col(0) - vertices.col(1)).norm(),
                              (vertices.col(1) - vertices.col(2)).norm(),
                              (vertices.col(2) - vertices.col(0)).norm()});
  // max modulus of velocity in quad nodes
  const double velocity_max =
      std::max_element(v_vals.begin(), v_vals.end(),
                       [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                         return a.norm() < b.norm();
                       })
          ->norm();
  const double kDelta = use_delta_ ? hK / (2.0 * velocity_max) : 1.0;
  // End compute the delta coefficient

  for (int l = 0; l < 6; ++l) {
    // Metric factor $\cob{|\det\Derv\Phibf_K(\wh{\zetabf}_{\ell})}$ scaled with
    // quadrature weight $\cob{\omega_{\ell}}$
    const double kFactor = qr_.Weights()[l] * dets[l];
    // Compute transformed gradients $\cob{\Vt_{\ell}^i}$ of all local shape
    // functions and collect them in the columns of a 2 x 6-matrix
    const Eigen::Matrix<double, 2, 6> kTrfGrad =
        JinvT.block(0, 2 * l, 2, 2) *
        (grad_ref_lsf_.block(0, 2 * l, 6, 2).transpose());
    // Compute the inner products of the transformed gradients with the
    // velocity vector at the current quadrature point.
    const Eigen::Matrix<double, 1, 6> mvec = v_vals[l].transpose() * kTrfGrad;
    // Advective part of the SUPG element matrix
    for (int j = 0; j < 6; ++j) {
      for (int i = 0; i < 6; ++i) {
        elem_mat(i, j) += kFactor * (kDelta * mvec[i] * mvec[j] +
                                     mvec[j] * val_ref_lsf_(i, l));
      }
    }
  }
  return elem_mat;
}

/**
 * @brief Solve the convection-diffusion boundary value problem using the SUPG
 * method for quadratic Lagrangian FE space
 * @tparam DIFFUSION_COEFF diffusion coefficient type
 * @tparam CONVECTION_COEFF convection coefficient type
 * @tparam FUNCTOR_F the source function type
 * @tparam FUNCTOR_G Dirichlet boundary function type
 * @param fe_space underlying FE space
 * @param eps epsilon parameter for the diffusive term
 * @param v velocity field
 * @param f source function
 * @param g Dirichlet boundary function
 * @param flagInflow flag to indicate what kind of inflow boundary we have, if
 * true we have rotational inflow, otherwise we have the standard inflow
 * @return solution vector
 */
template <typename DIFFUSION_COEFF, typename CONVECTION_COEFF,
          typename FUNCTOR_F, typename FUNCTOR_G>
Eigen::VectorXd SolveCDBVPSupgQuad(
    const std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO2<double>> &fe_space,
    DIFFUSION_COEFF eps, CONVECTION_COEFF v, FUNCTOR_F f, FUNCTOR_G g,
    bool flagInflow = false) {
  // Wrap functions into MeshFunctions
  lf::mesh::utils::MeshFunctionGlobal mf_eps{eps};
  lf::mesh::utils::MeshFunctionGlobal mf_v{v};
  lf::mesh::utils::MeshFunctionGlobal mf_f{f};
  lf::mesh::utils::MeshFunctionGlobal mf_g{g};

  auto mesh_p = fe_space->Mesh();
  const lf::assemble::DofHandler &dofh{fe_space->LocGlobMap()};

  lf::assemble::COOMatrix<double> A(dofh.NumDofs(), dofh.NumDofs());

  // Assemble Galerkin matrix
  // First the diffusive part
  lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider(
      fe_space, mf_eps, lf::mesh::utils::MeshFunctionConstant(0.0));
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider, A);

  // Diffusive part + Convective part - done based on the SUPG provider
  ecu_scheme::assemble::SUPGElementMatrixProvider supg_elmat_provider(mf_v,
                                                                      true);
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, supg_elmat_provider, A);

  // RHS vector
  Eigen::VectorXd phi(dofh.NumDofs());
  phi.setZero();
  lf::fe::ScalarLoadElementVectorProvider<double, decltype(mf_f)>
      load_element_vector_provider(fe_space, mf_f);
  lf::assemble::AssembleVectorLocally(0, dofh, load_element_vector_provider,
                                      phi);

  // IMPOSE DIRICHLET BOUNDARY CONDITIONS ON INFLOW BOUNDARY

  if (!flagInflow) {
    //    assemble::utils::EnforceBoundaryConditions(fe_space, A, phi, g);
    EnforceBoundaryConditions(fe_space, A, phi, g);
  } else {
    EnforceBoundaryConditionsOnRotInflow(fe_space, A, phi, g);
    //    ::utils::EnforceBoundaryConditionsOnRotInflow(fe_space, A, phi, g);
  }

  // SOLVE LINEAR SYSTEM
  Eigen::SparseMatrix<double> A_crs = A.makeSparse();
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_crs);
  if (solver.info() != Eigen::Success) {
    std::cerr << "LU decomposition failed for manufactured solution"
              << std::endl;
  }
  Eigen::VectorXd solution_vector = solver.solve(phi);
  if (solver.info() != Eigen::Success) {
    std::cerr << "Solving failed for manufactured solution" << std::endl;
  }
  return solution_vector;
}

}  // namespace ecu_scheme::assemble

#endif  // THESIS_ASSEMBLE_SUPG_MATRIX_PROVIDER_H_
