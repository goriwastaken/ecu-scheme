// Project header file include
// C system headers
// C++ standard library headers
// Other libraries headers

#include "const_velo_solution.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <memory>

namespace ecu_scheme::experiments{

//Eigen::VectorXd step(const Eigen::VectorXd &u0_vector,
//                     double tau,
//                     Eigen::SparseMatrix<double> convection_matrix_sparse) {
//  //Eigen::VectorXd result = Eigen::VectorXd::Zero(u0_vector.size());
//
//  // explicit midpoint rule computes solution to u'(t) = f(u(t)) with u(0) = u0
//  // u^* = u_n + tau/2 * f(u_n)
//  // u_{n+1} = u_n + tau * f(u^*)
//
//  // implement this for the pde d/dt u = -v * grad u
//  // get element matrix
//  LF_ASSERT_MSG(convection_matrix_sparse.rows() == convection_matrix_sparse.cols(), "Matrix must be square");
//  LF_ASSERT_MSG(convection_matrix_sparse.rows() == u0_vector.size(), "Matrix and vector must have the same size");
//
//  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
//  solver.compute(convection_matrix_sparse);
//  if(solver.info() != Eigen::Success){
//    std::cerr << "LU decomposition failed for const velo solution" << std::endl;
//  }
//  // Evaluate f(u_n)
//  Eigen::VectorXd k1 = solver.solve(u0_vector);
//  if(solver.info() != Eigen::Success){
//    std::cerr << "Solving failed for const velo solution" << std::endl;
//  }
//  // Evaluate f(u^*)
//  Eigen::VectorXd k2 = solver.solve(u0_vector + 0.5 * tau * k1);
////  solver.compute(lumped_mass_matrix_sparse);
////  Eigen::VectorXd k1 = solver.solve(convection_matrix_sparse * u0_vector);
////  Eigen::VectorXd k2 = solver.solve(convection_matrix_sparse * (u0_vector + 0.5 * tau * k1));
//
//  return u0_vector + tau * k2;
//}

void EnforceBoundaryConditions(const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> &fe_space,
                               lf::assemble::COOMatrix<double> &A,
                               Eigen::VectorXd &phi,
                               std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet
) {
  lf::mesh::utils::MeshFunctionGlobal<decltype(dirichlet)> mf_g_dirichlet{dirichlet};
  lf::mesh::utils::AllCodimMeshDataSet<bool> bd_flags(fe_space->Mesh(), false);
  // set a fixed epsilon value for the geometric test involving double comparison
  const double kTol = 1e-8;
  // Loop over all edges
  for (const auto& edge : fe_space->Mesh()->Entities(1)) {
    LF_ASSERT_MSG(edge->RefEl() == lf::base::RefEl::kSegment(),
                  "Entity should be an edge");
    const lf::geometry::Geometry* geo_ptr = edge->Geometry();
    const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
    // Check if the edge lies on $\Gamma_{\mathrm{in}}$  (geometric test)
    if ((corners(0, 0) + corners(0, 1)) / 2. < /*> 1. -*/ kTol ||
        (corners(1, 0) + corners(1, 1)) / 2. < kTol) {
      // Add the edge to the flagged entities
      bd_flags(*edge) = true;
    }
  }
  // Loop over all Points
  for (const auto& point : fe_space->Mesh()->Entities(2)) {
    LF_ASSERT_MSG(point->RefEl() == lf::base::RefEl::kPoint(),
                  "Entity should be an edge");
    const lf::geometry::Geometry* geo_ptr = point->Geometry();
    const Eigen::VectorXd coords = lf::geometry::Corners(*geo_ptr);
    // Check if the node lies on  $\Gamma_{\mathrm{in}}$ (geometric test)
    if (coords(0)  < /*> 1. -*/ kTol || coords(1) < kTol) {
      // Add the edge to the flagged entities
      bd_flags(*point) = true;
    }
  }
  auto flag_values{lf::fe::InitEssentialConditionFromFunction(
      *fe_space, bd_flags, mf_g_dirichlet)};

  lf::assemble::FixFlaggedSolutionCompAlt<double>(
      [&flag_values](lf::assemble::glb_idx_t dof_idx) {
        return flag_values[dof_idx];
      },
      A, phi);
}

} // experiments