// Project header file include
// C system headers
// C++ standard library headers
// Other libraries headers

#ifndef THESIS_EXPERIMENTS_CONCENTRIC_STREAM_CONCENTRIC_STREAM_SOLUTION_H_
#define THESIS_EXPERIMENTS_CONCENTRIC_STREAM_CONCENTRIC_STREAM_SOLUTION_H_

#include <Eigen/SVD>
#include <algorithm>
#include <memory>

#include "Eigen/Core"
#include "Eigen/QR"
#include "assemble.h"
#include "lf/assemble/assemble.h"
#include "lf/fe/fe.h"
#include "lf/io/io.h"
#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"

namespace ecu_scheme::experiments {

/**
 * @brief Enforces the Dirichlet boundary conditions at the inflow boundary
 * @tparam SCALAR template parameter indicating the scalar type
 * @param fe_space corresponding finite element space
 * @param A Galerkin system matrix
 * @param phi Right-hand side vector of the linear system
 * @param dirichlet Dirichlet function to be enforced at the inflow boundary
 */
void EnforceBoundaryConditions(
    const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> &fe_space,
    lf::assemble::COOMatrix<double> &A, Eigen::VectorXd &phi,
    std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet);

/**
 * @brief Flags all nodes on the inflow boundary - basically a different
 * implementation to do the same thing as EnforceBoundaryConditions
 * @param mesh_p underlying mesh
 * @param velocity velocity field
 * @return Data set of flags for nodes on the inflow boundary
 */
lf::mesh::utils::CodimMeshDataSet<bool> flagNodesOnInflowBoundary(
    const std::shared_ptr<const lf::mesh::Mesh> &mesh_p,
    std::function<
        Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double, 2, 1, 0> &)>
        velocity);

template <typename SCALAR>
class ConcentricStreamSolution {
 public:
  ConcentricStreamSolution() = delete;
  ~ConcentricStreamSolution() = default;

  explicit ConcentricStreamSolution(
      const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>>
          &fe_space)
      : fe_space_(fe_space) {}

  Eigen::Vector<double, Eigen::Dynamic> ComputeSolution(
      double eps,
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          velocity,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet)
      const {
    // Wrap all functors into LehrFEM MeshFunctions
    lf::mesh::utils::MeshFunctionConstant mf_eps{eps};
    lf::mesh::utils::MeshFunctionGlobal<decltype(velocity)> mf_velocity{
        velocity};
    lf::mesh::utils::MeshFunctionGlobal<decltype(dirichlet)> mf_g_dirichlet{
        dirichlet};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    // Diffusive part of the bilinear form is 0 -- pure transport problem
    lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider(
        fe_space_, lf::mesh::utils::MeshFunctionConstant(0.0),
        lf::mesh::utils::MeshFunctionConstant(0.0));
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

    // IMPOSE DIRICHLET BOUNDARY CONDITIONS
    // Obtain a predicate for all edges in the mesh
    //    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
    //    // Fetch flags and values for DOFs located on the boundary
    //    auto
    //    ess_bdc_flags_values{lf::fe::InitEssentialConditionFromFunction(*fe_space_,
    //    bd_flags, mf_g_dirichlet)};
    //    // Eliminate Dirichlet dofs from linear system
    //    lf::assemble::FixFlaggedSolutionComponents<double>(
    //        [&ess_bdc_flags_values](lf::assemble::glb_idx_t gdof_idx) {
    //          return ess_bdc_flags_values[gdof_idx];
    //        },
    //        A, phi);
    // IMPOSE DIRICHLET BOUNDARY CONDITIONS ON INFLOW BOUNDARY
    auto inflow_nodes{flagNodesOnInflowBoundary(mesh_p, velocity)};
    Eigen::VectorXd dirichlet_coefficients =
        lf::fe::NodalProjection(*fe_space_, mf_g_dirichlet);
    lf::assemble::FixFlaggedSolutionCompAlt<double>(
        [&inflow_nodes, &dirichlet_coefficients,
         &dofh](lf::assemble::glb_idx_t dof_idx) -> std::pair<bool, double> {
          const lf::mesh::Entity &dof_node{dofh.Entity(dof_idx)};
          LF_ASSERT_MSG(dof_node.RefEl() == lf::base::RefEl::kPoint(),
                        "All dofs must be associated with points");
          return {inflow_nodes(dof_node), dirichlet_coefficients[dof_idx]};
        },
        A, phi);

    //    EnforceBoundaryConditions(fe_space_, A, phi, dirichlet);

    // SOLVE LINEAR SYSTEM
    Eigen::SparseMatrix<double> A_crs = A.makeSparse();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_crs);
    if (solver.info() != Eigen::Success) {
      LF_VERIFY_MSG(false, "LU decomposition failed");
    }
    Eigen::VectorXd solution_vector = solver.solve(phi);
    if (solver.info() != Eigen::Success) {
      LF_VERIFY_MSG(false, "Solving failed");
    }
    return solution_vector;
  }  // end ComputeSolution
  /**
   * @brief Compute the solution for the given problem using multiple methods
   * @param eps
   * @param velocity
   * @param dirichlet
   * @param method_name should be either {"UPWIND", "STABLE_UPWIND",
   * "15P_UPWIND"}
   * @return
   */
  Eigen::Vector<double, Eigen::Dynamic> ComputeSolutionMultipleMethods(
      double eps,
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          velocity,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet,
      std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)>
          test_functor,
      std::string method_name) const {
    // Wrap all functors into LehrFEM MeshFunctions
    lf::mesh::utils::MeshFunctionConstant mf_eps{eps};
    lf::mesh::utils::MeshFunctionGlobal<decltype(velocity)> mf_velocity{
        velocity};
    lf::mesh::utils::MeshFunctionGlobal<decltype(dirichlet)> mf_g_dirichlet{
        dirichlet};
    lf::mesh::utils::MeshFunctionGlobal<decltype(test_functor)> mf_f_test{
        test_functor};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    // Diffusive part of the bilinear form is 0 -- pure transport problem
    //    lf::uscalfe::ReactionDiffusionElementMatrixProvider
    //    laplacian_provider(
    //        fe_space_, lf::mesh::utils::MeshFunctionConstant(0.0),
    //        lf::mesh::utils::MeshFunctionConstant(0.0)
    //    );
    //    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider,
    //    A);

    // Convective part of the bilinear form - done based on ecu_scheme::assemble
    if (fe_space_->LocGlobMap().NumLocalDofs(*(mesh_p->EntityByIndex(0, 0))) ==
        6) {
      if (method_name == "UPWIND") {
        ecu_scheme::assemble::ConvectionUpwindMatrixProvider<SCALAR,
                                                             decltype(velocity)>
            convection_provider(
                fe_space_, velocity,
                ecu_scheme::assemble::initializeMassesQuadratic(mesh_p),
                ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
        lf::assemble::AssembleMatrixLocally(0, dofh, dofh, convection_provider,
                                            A);
      } else if (method_name == "STABLE_UPWIND") {
        ecu_scheme::assemble::StableConvectionUpwindMatrixProvider<
            SCALAR, decltype(velocity)>
            stable_convection_provider(
                fe_space_, velocity,
                ecu_scheme::assemble::initMassesVert(mesh_p),
                ecu_scheme::assemble::initMassesEdges(mesh_p),
                ecu_scheme::assemble::initMassesCells(mesh_p));
        lf::assemble::AssembleMatrixLocally(0, dofh, dofh,
                                            stable_convection_provider, A);
      } else if (method_name == "15P_UPWIND") {
        ecu_scheme::assemble::FifteenPointUpwindMatrixProvider<
            SCALAR, decltype(velocity)>
            fifteen_point_upwind_matrix_provider(
                fe_space_, velocity,
                ecu_scheme::assemble::initMassesVerticesFifteenQuadRule(mesh_p),
                ecu_scheme::assemble::initMassesEdgeMidpointsFifteenQuadRule(
                    mesh_p),
                ecu_scheme::assemble::initMassesEdgeOffFifteenQuadRule(mesh_p),
                ecu_scheme::assemble::initMassesCellsFifteenQuadRule(mesh_p));
        lf::assemble::AssembleMatrixLocally(
            0, dofh, dofh, fifteen_point_upwind_matrix_provider, A);
      } else {
        std::cerr << "Invalid method name" << std::endl;
      }

    } else if (fe_space_->LocGlobMap().NumLocalDofs(
                   *(mesh_p->EntityByIndex(0, 0))) == 3) {
      if (method_name == "UPWIND") {
        ecu_scheme::assemble::ConvectionUpwindMatrixProvider<SCALAR,
                                                             decltype(velocity)>
            convection_provider(
                fe_space_, velocity,
                ecu_scheme::assemble::initializeMasses(mesh_p),
                ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
        lf::assemble::AssembleMatrixLocally(0, dofh, dofh, convection_provider,
                                            A);
      } else if (method_name == "STABLE_UPWIND") {
        std::cerr << "Only implemented for quadratic FE space" << std::endl;
      } else if (method_name == "15P_UPWIND") {
        std::cerr << "Only implemented for quadratic FE space" << std::endl;
      } else {
        std::cerr << "Invalid method name" << std::endl;
      }
    }

    // Right-hand side vector
    Eigen::VectorXd phi(N_dofs);
    phi.setZero();
    // Assemble RHS
    lf::uscalfe::ScalarLoadElementVectorProvider elvec_provider(fe_space_,
                                                                mf_f_test);
    lf::assemble::AssembleVectorLocally(0, dofh, elvec_provider, phi);

    //    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> check_rank(A.makeDense());
    //    auto rank = check_rank.rank();
    //    auto det_val = check_rank.absDeterminant();
    // IMPOSE DIRICHLET BOUNDARY CONDITIONS ON INFLOW BOUNDARY
    EnforceBoundaryConditions(fe_space_, A, phi, dirichlet);

    // uncomment for debug smallest singular value of matrix
    //    if(method_name == "15P_UPWIND"){
    //      Eigen::MatrixXd A_svd = A.makeDense();
    //      std::cout << A_svd.rows() << " cols: " << A_svd.cols() << std::endl;
    //      Eigen::BDCSVD<Eigen::MatrixXd> svd(A_svd,  Eigen::ComputeFullV);
    //      std::cout << "SVD Computation: " << std::endl;
    ////      Eigen::MatrixXd U = svd.matrixU();
    //      Eigen::MatrixXd V = svd.matrixV();
    ////      Eigen::VectorXd sv = svd.singularValues();
    ////      Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(A_svd.rows(),
    ///A_svd.cols()); /      const unsigned int p = sv.size(); /
    ///Sigma.block(0,0,p,p) = sv.asDiagonal();
    //      // U, Sigma, V tuple
    //      const unsigned int last_col_sing_vector = V.cols() - 1;
    //      Eigen::VectorXd smallest_singular_vector =
    //      V.col(last_col_sing_vector);
    //      //output results to vtk file quadratic lagrangian - singular vector
    //      lf::fe::MeshFunctionFE mf_sol(fe_space_, smallest_singular_vector);
    //      lf::io::VtkWriter vtk_writer(fe_space_->Mesh(), PROJECT_BUILD_DIR
    //      "/results/concentric_stream_singular_vector.vtk", 0, 2);
    //      vtk_writer.WritePointData("_singular_vector", mf_sol);
    //
    //    }

    // SOLVE LINEAR SYSTEM
    Eigen::SparseMatrix<double> A_crs = A.makeSparse();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_crs);
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
};

}  // namespace ecu_scheme::experiments

#endif  // THESIS_EXPERIMENTS_CONCENTRIC_STREAM_CONCENTRIC_STREAM_SOLUTION_H_
