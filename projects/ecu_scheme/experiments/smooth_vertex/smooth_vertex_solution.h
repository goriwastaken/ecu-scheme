
#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_EXPERIMENTS_SMOOTH_VERTEX_SMOOTH_VERTEX_SOLUTION_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_EXPERIMENTS_SMOOTH_VERTEX_SMOOTH_VERTEX_SOLUTION_H_

#include <functional>
#include <utility>

#include "lf/assemble/assemble.h"
#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"
#include "lf/fe/fe.h"

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "Eigen/SparseLU"
#include <memory>

#include "assemble.h"

namespace ecu_scheme::experiments {

template <typename SCALAR>
class SmoothVertexSolution {
 public:
  SmoothVertexSolution() = default;
  ~SmoothVertexSolution() = default;
  SmoothVertexSolution(const SmoothVertexSolution &other) = default;
  SmoothVertexSolution(SmoothVertexSolution &&other) noexcept = default;
  SmoothVertexSolution &operator=(const SmoothVertexSolution &other) = default;
  SmoothVertexSolution &operator=(SmoothVertexSolution &&other) noexcept =
      default;

  explicit SmoothVertexSolution(
      const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>>
          &fe_space)
      : fe_space_(fe_space) {}

  std::vector<Eigen::VectorXd> ComputeSolution(
      std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double, 3, 1, 0>&)> velocity,
      std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double, 3, 1, 0>&)> magnetic_induction,
      std::function<double(const Eigen::Matrix<double, 3, 1, 0>&)> potential,
      const double max_time,
      const double time_step
      ) const {

    // Wrap all functors into LehrFEM MeshFunctions
    lf::mesh::utils::MeshFunctionGlobal mf_velocity{velocity};
    lf::mesh::utils::MeshFunctionGlobal mf_magnetic_induction{magnetic_induction};
    lf::mesh::utils::MeshFunctionGlobal mf_potential{potential};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    if(fe_space_->LocGlobMap().NumLocalDofs(*(mesh_p->EntityByIndex(0,0))) == 6) {
      // Quadratic FE space
      ecu_scheme::assemble::ConvectionUpwindMatrixProvider<double, decltype(velocity)> convection_upwind_provider(fe_space_, velocity, ecu_scheme::assemble::initializeMassesQuadratic(mesh_p), ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh, convection_upwind_provider, A);
    }else if(fe_space_->LocGlobMap().NumLocalDofs(*(mesh_p->EntityByIndex(0,0))) == 3) {
      // Linear FE space
      ecu_scheme::assemble::ConvectionUpwindMatrixProvider<double, decltype(velocity)> convection_upwind_provider(fe_space_, velocity, ecu_scheme::assemble::initializeMasses(mesh_p), ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh,
                                          convection_upwind_provider, A);
    }

    // RHS vector
    Eigen::VectorXd phi(N_dofs);
    phi.setZero();

    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
    //todo finish boundary conditions
    // the initial condition is the potential at time t = 0
    const auto initial_condition = [potential](const Eigen::Vector2d& xh)-> double{
      const Eigen::Vector3d temp(xh(0), xh(1), 0.0);
      return potential(temp);
    };
    lf::mesh::utils::MeshFunctionGlobal mf_initial_condition{initial_condition};

    auto ess_bdc_flags_values{lf::fe::InitEssentialConditionFromFunction(*fe_space_, bd_flags, lf::mesh::utils::MeshFunctionConstant(0.0))};
    lf::assemble::FixFlaggedSolutionComponents<double>(
        [&ess_bdc_flags_values](lf::assemble::glb_idx_t gdof_idx){
          return ess_bdc_flags_values[gdof_idx];
        },
        A, phi);


    // Solve the transient advection problem
    Eigen::SparseMatrix<double> convection_matrix_sparse = A.makeSparse();
    std::vector<Eigen::VectorXd> solution_vector;

    solution_vector.push_back(lf::fe::NodalProjection(*fe_space_, mf_initial_condition));

    // Runge-Kutta 2nd order midpoint rule
    for(int iter = 0; iter < N_dofs - 1; ++iter){
      Eigen::VectorXd current_solution = solution_vector.at(iter);
      // step of RK-SSM
      // explicit midpoint rule computes solution to u'(t) = f(u(t)) with u(0) = u0
      // u^* = u_n + tau/2 * f(u_n)
      // u_{n+1} = u_n + tau * f(u^*)
      Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
      solver.compute(convection_matrix_sparse);
      if(solver.info() != Eigen::Success){
        std::cerr << "LU decomposition failed for smooth vertex solution" << std::endl;
      }
      // Evaluate f(u_n)
      Eigen::VectorXd k1 = solver.solve(current_solution);
      if(solver.info() != Eigen::Success){
        std::cerr << "Solving failed for const velo solution" << std::endl;
      }
      // Evaluate f(u^*)
      Eigen::VectorXd k2 = solver.solve(current_solution + 0.5 * time_step * k1);

      solution_vector.push_back(current_solution + time_step * k2);
    }
    return solution_vector;
  }

 private:
  std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space_;
};

}  // namespace ecu_scheme

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_EXPERIMENTS_SMOOTH_VERTEX_SMOOTH_VERTEX_SOLUTION_H_
