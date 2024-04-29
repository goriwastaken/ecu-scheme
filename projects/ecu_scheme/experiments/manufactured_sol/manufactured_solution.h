#ifndef THESIS_EXPERIMENTS_MANUFACTURED_SOL_EXP_MANUFACTURED_SOLUTION_H_
#define THESIS_EXPERIMENTS_MANUFACTURED_SOL_EXP_MANUFACTURED_SOLUTION_H_

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <functional>
#include <memory>
#include <utility>

#include "assemble.h"
#include "lf/assemble/assemble.h"
#include "lf/fe/fe.h"
#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"

namespace ecu_scheme::experiments {

void EnforceBoundaryConditions(
    const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> &fe_space,
    lf::assemble::COOMatrix<double> &A, Eigen::VectorXd &phi,
    std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet);

/**
 * @brief Class for computing the manufactured solution for the ECU scheme
 */
template <typename SCALAR>
class ManufacturedSolutionExperiment {
 public:
  ManufacturedSolutionExperiment() = delete;
  ~ManufacturedSolutionExperiment() = default;
  ManufacturedSolutionExperiment(const ManufacturedSolutionExperiment &other) =
      default;
  ManufacturedSolutionExperiment(
      ManufacturedSolutionExperiment &&other) noexcept = default;
  ManufacturedSolutionExperiment &operator=(
      const ManufacturedSolutionExperiment &other) = default;
  ManufacturedSolutionExperiment &operator=(
      ManufacturedSolutionExperiment &&other) noexcept = default;

  //    explicit ManufacturedSolutionExperiment(std::shared_ptr<lf::mesh::Mesh>
  //    mesh)
  //        : mesh_(std::move(mesh)) {}

  explicit ManufacturedSolutionExperiment(
      const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>>
          &fe_space)
      : fe_space_(fe_space) {}

  /**
   * @brief Computes the manufactured solution for the ECU scheme
   * @param eps Epsilon diffusion coefficient
   * @param velocity functor for the velocity field
   * @param test test function f
   * @param dirichlet dirichlet function g
   * @return
   */
  Eigen::Vector<double, Eigen::Dynamic> ComputeSolution(
      double eps,
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          velocity,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> test,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet)
      const {
    // Wrap all functors into LehrFEM MeshFunctions
    const auto kTempEps = [eps](const Eigen::Vector2d &x) { return eps; };
    lf::mesh::utils::MeshFunctionGlobal<decltype(kTempEps)> mf_eps{kTempEps};
    //      lf::mesh::utils::MeshFunctionConstant mf_eps{eps};
    lf::mesh::utils::MeshFunctionGlobal<decltype(velocity)> mf_velocity{
        velocity};
    lf::mesh::utils::MeshFunctionGlobal<decltype(test)> mf_f_test_function{
        test};
    lf::mesh::utils::MeshFunctionGlobal<decltype(dirichlet)> mf_g_dirichlet{
        dirichlet};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    // Diffusive part of the bilinear form
    lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider(
        fe_space_, mf_eps, lf::mesh::utils::MeshFunctionConstant(0.0));
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider, A);
    // Convective part of the bilinear form - done based on ecu_scheme::assemble
    if (fe_space_->LocGlobMap().NumLocalDofs(*(mesh_p->EntityByIndex(0, 0))) ==
        6) {
      ecu_scheme::assemble::ConvectionUpwindMatrixProvider<SCALAR,
                                                           decltype(velocity)>
          convection_provider(
              fe_space_, velocity,
              ecu_scheme::assemble::initializeMassesQuadratic(mesh_p),
              ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh, convection_provider,
                                          A);
    } else if (fe_space_->LocGlobMap().NumLocalDofs(
                   *(mesh_p->EntityByIndex(0, 0))) == 3) {
      ecu_scheme::assemble::ConvectionUpwindMatrixProvider<SCALAR,
                                                           decltype(velocity)>
          convection_provider(
              fe_space_, velocity,
              ecu_scheme::assemble::initializeMasses(mesh_p),
              ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh, convection_provider,
                                          A);
    }

    // Right-hand side vector
    Eigen::VectorXd phi(N_dofs);
    phi.setZero();
    lf::fe::ScalarLoadElementVectorProvider<double,
                                            decltype(mf_f_test_function)>
        load_provider(fe_space_, mf_f_test_function);
    lf::assemble::AssembleVectorLocally(0, dofh, load_provider, phi);
    // IMPOSE DIRICHLET CONDITIONS:
    // Create boundary flags
    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
    // Fetch flags and values for DOFs located on the boundary
    auto ess_bdc_flags_values{lf::fe::InitEssentialConditionFromFunction(
        *fe_space_, bd_flags, mf_g_dirichlet)};
    // Eliminate Dirichlet dofs from linear system
    lf::assemble::FixFlaggedSolutionComponents<double>(
        [&ess_bdc_flags_values](lf::uscalfe::glb_idx_t gdof_idx) {
          return ess_bdc_flags_values[gdof_idx];
        },
        A, phi);
    //      EnforceBoundaryConditions(fe_space_, A, phi, dirichlet);

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
  }  // end ComputeSolution

  /**
   * @brief Solves the Convection-Diffusion BVP with nonhomogeneous Dirichlet
   * boundary conditions using the SUPG method and quadratic FE space
   * @param eps Epsilon diffusion coefficient
   * @param velocity functor for the velocity field
   * @param test test function f
   * @param dirichlet dirichlet boundary condition function g
   * @return Eigen solution vector
   */
  Eigen::Vector<double, Eigen::Dynamic> ComputeSUPGSolution(
      double eps,
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          velocity,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> test,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet)
      const {
    // Wrap all functors into LehrFEM MeshFunctions
    const auto kTempEps = [eps](const Eigen::Vector2d &x) { return eps; };
    lf::mesh::utils::MeshFunctionGlobal<decltype(kTempEps)> mf_eps{kTempEps};
    lf::mesh::utils::MeshFunctionGlobal mf_velocity{velocity};
    lf::mesh::utils::MeshFunctionGlobal<decltype(test)> mf_f_test_function{
        test};
    lf::mesh::utils::MeshFunctionGlobal<decltype(dirichlet)> mf_g_dirichlet{
        dirichlet};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    // Diffusive part of the bilinear form
    lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider(
        fe_space_, mf_eps, lf::mesh::utils::MeshFunctionConstant(0.0));
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider, A);

    // Convective part of the bilinear form - done based on the SUPG provider
    // from ecu_scheme::assemble
    ecu_scheme::assemble::SUPGElementMatrixProvider supg_elmat_provider(
        mf_velocity, true);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, supg_elmat_provider, A);

    // Right-hand side vector
    Eigen::VectorXd phi(N_dofs);
    phi.setZero();
    lf::fe::ScalarLoadElementVectorProvider<double,
                                            decltype(mf_f_test_function)>
        load_provider(fe_space_, mf_f_test_function);
    lf::assemble::AssembleVectorLocally(0, dofh, load_provider, phi);

    // IMPOSE DIRICHLET CONDITIONS:
    // Create boundary flags
    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
    // Fetch flags and values for DOFs located on the boundary
    auto ess_bdc_flags_values{lf::fe::InitEssentialConditionFromFunction(
        *fe_space_, bd_flags, mf_g_dirichlet)};
    // Eliminate Dirichlet dofs from linear system
    lf::assemble::FixFlaggedSolutionComponents<double>(
        [&ess_bdc_flags_values](lf::uscalfe::glb_idx_t gdof_idx) {
          return ess_bdc_flags_values[gdof_idx];
        },
        A, phi);
    //      EnforceBoundaryConditions(fe_space_, A, phi, dirichlet);

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
  }  // end ComputeSUPGSolution

  /**
   * @brief Implemented only for quadratic FE space
   * @param eps
   * @param velocity
   * @param test
   * @param dirichlet
   * @return
   */
  Eigen::Vector<double, Eigen::Dynamic> ComputeStableSolution(
      double eps,
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          velocity,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> test,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet)
      const {
    const auto kTempEps = [eps](const Eigen::Vector2d &x) { return eps; };
    lf::mesh::utils::MeshFunctionGlobal<decltype(kTempEps)> mf_eps{kTempEps};
    lf::mesh::utils::MeshFunctionGlobal mf_velocity{velocity};
    lf::mesh::utils::MeshFunctionGlobal<decltype(test)> mf_f_test_function{
        test};
    lf::mesh::utils::MeshFunctionGlobal<decltype(dirichlet)> mf_g_dirichlet{
        dirichlet};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    // Diffusive part
    lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider(
        fe_space_, mf_eps, lf::mesh::utils::MeshFunctionConstant(0.0));
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider, A);

    // Convective part - using the Stable scheme b^{2b, vmb} from HH08
    ecu_scheme::assemble::StableConvectionUpwindMatrixProvider<
        SCALAR, decltype(velocity)>
        stable_elmat_provider(fe_space_, velocity,
                              ecu_scheme::assemble::initMassesVert(mesh_p),
                              ecu_scheme::assemble::initMassesEdges(mesh_p),
                              ecu_scheme::assemble::initMassesCells(mesh_p));
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, stable_elmat_provider,
                                        A);

    // Right-hand side vector
    Eigen::VectorXd phi(N_dofs);
    phi.setZero();
    lf::fe::ScalarLoadElementVectorProvider<double,
                                            decltype(mf_f_test_function)>
        load_provider(fe_space_, mf_f_test_function);
    lf::assemble::AssembleVectorLocally(0, dofh, load_provider, phi);
    // IMPOSE DIRICHLET CONDITIONS:
    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
    auto ess_bdc_flags_values{lf::fe::InitEssentialConditionFromFunction(
        *fe_space_, bd_flags, mf_g_dirichlet)};
    lf::assemble::FixFlaggedSolutionComponents<double>(
        [&ess_bdc_flags_values](lf::uscalfe::glb_idx_t gdof_idx) {
          return ess_bdc_flags_values[gdof_idx];
        },
        A, phi);
    //      EnforceBoundaryConditions(fe_space_, A, phi, dirichlet);

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
  }  // end ComputeStableSolution

  Eigen::Vector<double, Eigen::Dynamic> ComputeFifteenPointQuadRuleSolution(
      double eps,
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          velocity,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> test,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet)
      const {
    const auto kTempEps = [eps](const Eigen::Vector2d &x) { return eps; };
    lf::mesh::utils::MeshFunctionGlobal<decltype(kTempEps)> mf_eps{kTempEps};
    lf::mesh::utils::MeshFunctionGlobal mf_velocity{velocity};
    lf::mesh::utils::MeshFunctionGlobal<decltype(test)> mf_f_test_function{
        test};
    lf::mesh::utils::MeshFunctionGlobal<decltype(dirichlet)> mf_g_dirichlet{
        dirichlet};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    // Diffusive part
    lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider(
        fe_space_, mf_eps, lf::mesh::utils::MeshFunctionConstant(0.0));
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider, A);

    // Convective part - using the Stable scheme b^{2b, vmb} from HH08
    ecu_scheme::assemble::FifteenPointUpwindMatrixProvider<SCALAR,
                                                           decltype(velocity)>
        fifteen_point_elmat_provider(
            fe_space_, velocity,
            ecu_scheme::assemble::initMassesVerticesFifteenQuadRule(mesh_p),
            ecu_scheme::assemble::initMassesEdgeMidpointsFifteenQuadRule(
                mesh_p),
            ecu_scheme::assemble::initMassesEdgeOffFifteenQuadRule(mesh_p),
            ecu_scheme::assemble::initMassesCellsFifteenQuadRule(mesh_p));
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh,
                                        fifteen_point_elmat_provider, A);

    // Right-hand side vector
    Eigen::VectorXd phi(N_dofs);
    phi.setZero();
    lf::fe::ScalarLoadElementVectorProvider<double,
                                            decltype(mf_f_test_function)>
        load_provider(fe_space_, mf_f_test_function);
    lf::assemble::AssembleVectorLocally(0, dofh, load_provider, phi);
    // IMPOSE DIRICHLET CONDITIONS:
    //      auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
    //      auto
    //      ess_bdc_flags_values{lf::fe::InitEssentialConditionFromFunction(*fe_space_,
    //      bd_flags, mf_g_dirichlet)};
    //      lf::assemble::FixFlaggedSolutionComponents<double>(
    //          [&ess_bdc_flags_values](lf::uscalfe::glb_idx_t gdof_idx){
    //            return ess_bdc_flags_values[gdof_idx];
    //          }, A, phi);
    EnforceBoundaryConditions(fe_space_, A, phi, dirichlet);

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
  }  // end ComputeFifteenPointQuadRuleSolution

 private:
  std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space_;
  //  std::shared_ptr<lf::mesh::Mesh> mesh_;
};

}  // namespace ecu_scheme::experiments

#endif  // THESIS_EXPERIMENTS_MANUFACTURED_SOL_EXP_MANUFACTURED_SOLUTION_H_
