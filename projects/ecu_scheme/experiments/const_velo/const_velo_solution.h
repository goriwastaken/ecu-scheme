// Project header file include
// C system headers
// C++ standard library headers
// Other libraries headers

#ifndef THESIS_EXPERIMENTS_CONST_VELO_CONST_VELO_SOLUTION_H_
#define THESIS_EXPERIMENTS_CONST_VELO_CONST_VELO_SOLUTION_H_

#include <functional>
#include <memory>
#include <utility>

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "Eigen/SparseLU"
#include "assemble.h"
#include "lf/assemble/assemble.h"
#include "lf/fe/fe.h"
#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"

namespace ecu_scheme::experiments {

/**
 * @brief Class for generating the Constant Velocity experiment setup and
 * computing the solution
 * @tparam SCALAR the scalar type of an underlying Lagrangian finite element
 * space
 */
template <typename SCALAR>
class ConstVeloSolution {
 public:
  ConstVeloSolution() = default;
  ~ConstVeloSolution() = default;
  ConstVeloSolution(const ConstVeloSolution &other) = default;
  ConstVeloSolution(ConstVeloSolution &&other) noexcept = default;
  ConstVeloSolution &operator=(const ConstVeloSolution &other) = default;
  ConstVeloSolution &operator=(ConstVeloSolution &&other) noexcept = default;

  explicit ConstVeloSolution(
      const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>>
          &fe_space)
      : fe_space_(fe_space) {}

  /**
   * @brief Computes the solution vector of the Constant Velocity experiment
   * @param dirichlet Dirichlet boundary condition
   * @param velocity velocity field assumed to be of the form \f$ (c,c)^T \f$
   * where \f$c\f$ is some constant
   * @param test_function source function
   * @param method_name should be either "UPWIND" or "STABLE_UPWIND"
   * representing the midpoint and 7-point upwind scheme respectively
   * @return vector of solution coefficients
   */
  Eigen::VectorXd ComputeSolution(
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet,
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          velocity,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)>
          test_function,
      std::string method_name) const {
    // Wrap all functors into LehrFEM MeshFunctions
    lf::mesh::utils::MeshFunctionGlobal mf_dirichlet{dirichlet};
    lf::mesh::utils::MeshFunctionGlobal mf_velocity{velocity};
    lf::mesh::utils::MeshFunctionGlobal mf_test_function{test_function};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> B(N_dofs, N_dofs);

    // Assemble convection matrix
    if (method_name == "UPWIND") {
      ecu_scheme::assemble::ConvectionUpwindMatrixProvider<double,
                                                           decltype(velocity)>
          convection_upwind_provider(
              fe_space_, velocity,
              ecu_scheme::assemble::initializeMassesQuadratic(mesh_p),
              ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh,
                                          convection_upwind_provider, B);
    } else if (method_name == "STABLE_UPWIND") {
      ecu_scheme::assemble::StableConvectionUpwindMatrixProvider<
          double, decltype(velocity)>
          convection_stable_upwind_provider(
              fe_space_, velocity, ecu_scheme::assemble::initMassesVert(mesh_p),
              ecu_scheme::assemble::initMassesEdges(mesh_p),
              ecu_scheme::assemble::initMassesCells(mesh_p));
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh,
                                          convection_stable_upwind_provider, B);
    }
    // RHS vector
    Eigen::VectorXd phi(N_dofs);
    phi.setZero();
    lf::fe::ScalarLoadElementVectorProvider<double, decltype(mf_test_function)>
        load_provider(fe_space_, mf_test_function);
    lf::assemble::AssembleVectorLocally(0, dofh, load_provider, phi);

    //    EnforceBoundaryConditions(fe_space_, B, phi, dirichlet);
    ecu_scheme::assemble::EnforceBoundaryConditions(fe_space_, B, phi,
                                                    dirichlet);

    Eigen::SparseMatrix<double> B_crs = B.makeSparse();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(B_crs);
    if (solver.info() != Eigen::Success) {
      LF_VERIFY_MSG(false, "LU decomposition failed");
    }
    Eigen::VectorXd solution_vector = solver.solve(phi);
    if (solver.info() != Eigen::Success) {
      LF_VERIFY_MSG(false, "Solving failed");
    }

    return solution_vector;
  }

 private:
  std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space_;
  //  Eigen::SparseMatrix<double> convection_matrix_sparse_;
};

}  // namespace ecu_scheme::experiments

#endif  // THESIS_EXPERIMENTS_CONST_VELO_CONST_VELO_SOLUTION_H_
