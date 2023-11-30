#ifndef THESIS_EXPERIMENTS_MANUFACTURED_SOL_EXP_MANUFACTURED_SOLUTION_H_
#define THESIS_EXPERIMENTS_MANUFACTURED_SOL_EXP_MANUFACTURED_SOLUTION_H_

#include <functional>
#include <utility>

#include "lf/assemble/assemble.h"
#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"
#include "lf/fe/fe.h"

#include "Eigen/Core"
#include <Eigen/SparseLU>

#include "assemble.h"

namespace ecu_scheme::experiments {

/**
 * @brief Class for computing the manufactured solution for the ECU scheme
 */
template <typename SCALAR>
class ManufacturedSolutionExperiment {
 public:
    ManufacturedSolutionExperiment() = delete;
    ~ManufacturedSolutionExperiment() = default;

//    explicit ManufacturedSolutionExperiment(std::shared_ptr<lf::mesh::Mesh> mesh)
//        : mesh_(std::move(mesh)) {}

    explicit ManufacturedSolutionExperiment(std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space)
        : fe_space_(std::move(fe_space)) {}

    /**
     * @brief Computes the manufactured solution for the ECU scheme
     * @param eps Epsilon diffusion coefficient
     * @param velocity functor for the velocity field
     * @param test test function f
     * @param dirichlet dirichlet function g
     * @return
     */
    Eigen::Vector<double, Eigen::Dynamic> ComputeSolution(double eps,
                                                          std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double,2,1,0>&)> velocity,
                                                          std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> test,
                                                          std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet
                                                          ) const {
      // Wrap all functors into LehrFEM MeshFunctions
      lf::mesh::utils::MeshFunctionConstant mf_eps{-eps};
      lf::mesh::utils::MeshFunctionGlobal<decltype(velocity)> mf_velocity{velocity};
      lf::mesh::utils::MeshFunctionGlobal<decltype(test)> mf_f_test_function{test};
      lf::mesh::utils::MeshFunctionGlobal<decltype(dirichlet)> mf_g_dirichlet{dirichlet};

      auto mesh_p = fe_space_->Mesh();
      const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
      const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
      lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

      // Diffusive part of the bilinear form
      lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider(
          fe_space_, mf_eps, lf::mesh::utils::MeshFunctionConstant(0.0)
      );
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider, A);
      // Convective part of the bilinear form - done based on ecu_scheme::assemble
      ecu_scheme::assemble::ConvectionUpwindMatrixProvider<SCALAR, decltype(velocity)> convection_provider(fe_space_, velocity, ecu_scheme::assemble::initializeMassesQuadratic(mesh_p), ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh, convection_provider, A);

      // Right-hand side vector
      Eigen::VectorXd phi(N_dofs);
      lf::fe::ScalarLoadElementVectorProvider<double, decltype(mf_f_test_function)> load_provider(fe_space_, mf_f_test_function);
      lf::assemble::AssembleVectorLocally(0, dofh, load_provider, phi);

      // IMPOSE DIRICHLET CONDITIONS:
      // Create boundary flags
      auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
      // Fetch flags and values for DOFs located on the boundary
      auto ess_bdc_flags_values{lf::fe::InitEssentialConditionFromFunction(*fe_space_, bd_flags, mf_g_dirichlet)};
      // Eliminate Dirichlet dofs from linear system
      lf::assemble::FixFlaggedSolutionComponents<double>(
          [&ess_bdc_flags_values](lf::uscalfe::glb_idx_t gdof_idx){
            return ess_bdc_flags_values[gdof_idx];
          }, A, phi);

      // SOLVE LINEAR SYSTEM
      Eigen::SparseMatrix A_crs = A.makeSparse();
      Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
      solver.compute(A_crs);
      if(solver.info() != Eigen::Success){
        std::cerr << "LU decomposition failed for manufactured solution" << std::endl;
      }
      Eigen::VectorXd solution_vector = solver.solve(phi);
      if(solver.info() != Eigen::Success){
        std::cerr << "Solving failed for manufactured solution" << std::endl;
      }
      return solution_vector;
    } // end ComputeSolution

 private:
  std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space_;
//  std::shared_ptr<lf::mesh::Mesh> mesh_;
};

} // experiments

#endif //THESIS_EXPERIMENTS_MANUFACTURED_SOL_EXP_MANUFACTURED_SOLUTION_H_
