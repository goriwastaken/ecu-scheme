// Project header file include
// C system headers
// C++ standard library headers
// Other libraries headers

#ifndef THESIS_EXPERIMENTS_CONCENTRIC_STREAM_CONCENTRIC_STREAM_SOLUTION_H_
#define THESIS_EXPERIMENTS_CONCENTRIC_STREAM_CONCENTRIC_STREAM_SOLUTION_H_

#include <algorithm>
#include <memory>

#include "lf/assemble/assemble.h"
#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"
#include "lf/fe/fe.h"

#include "Eigen/Core"

#include "assemble.h"

namespace ecu_scheme::experiments {


template <typename SCALAR>
class ConcentricStreamSolution{
 public:
  ConcentricStreamSolution() = delete;
  ~ConcentricStreamSolution() = default;

  explicit ConcentricStreamSolution(std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space)
      : fe_space_(std::move(fe_space)) {}


  Eigen::Vector<double, Eigen::Dynamic> ComputeSolution(double eps,
                                                        std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double, 2, 1, 0> &)> velocity,
                                                        std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet
                                                            ) const {
    // Wrap all functors into LehrFEM MeshFunctions
    lf::mesh::utils::MeshFunctionConstant mf_eps{eps};
    lf::mesh::utils::MeshFunctionGlobal<decltype(velocity)> mf_velocity{velocity};
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
    // Convective part of the bilinear form
    ecu_scheme::assemble::ConvectionUpwindMatrixProvider<SCALAR, decltype(velocity)> convection_provider(fe_space_, velocity, ecu_scheme::assemble::initializeMassesQuadratic(mesh_p), ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, convection_provider, A);

    // Right-hand side vector
    Eigen::VectorXd phi(N_dofs);
    phi.setZero();

    // IMPOSE DIRICHLET BOUNDARY CONDITIONS
    // Obtain a predicate for all edges in the mesh
    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
    // Fetch flags and values for DOFs located on the boundary
    auto ess_bdc_flags_values{lf::fe::InitEssentialConditionFromFunction(*fe_space_, bd_flags, mf_g_dirichlet)};
    // Eliminate Dirichlet dofs from linear system
    lf::assemble::FixFlaggedSolutionComponents<double>(
        [&ess_bdc_flags_values](lf::assemble::glb_idx_t gdof_idx) {
          return ess_bdc_flags_values[gdof_idx];
        },
        A, phi);

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
  } // end ComputeSolution

 private:
  std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space_;
};

} // namespace ecu_scheme::experiments

#endif //THESIS_EXPERIMENTS_CONCENTRIC_STREAM_CONCENTRIC_STREAM_SOLUTION_H_
