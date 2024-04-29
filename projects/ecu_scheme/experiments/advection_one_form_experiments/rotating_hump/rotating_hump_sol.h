#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_EXPERIMENTS_ADVECTION_ONE_FORM_EXPERIMENTS_ROTATING_HUMP_ROTATING_HUMP_SOL_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_EXPERIMENTS_ADVECTION_ONE_FORM_EXPERIMENTS_ROTATING_HUMP_ROTATING_HUMP_SOL_H_

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

/**
 * @brief Enforces the Dirichlet boundary conditions at the inflow boundary
 * @tparam SCALAR template parameter indicating the scalar type
 * @param fe_space corresponding finite element space
 * @param A Galerkin system matrix
 * @param phi Right-hand side vector of the linear system
 * @param dirichlet Dirichlet function to be enforced at the inflow boundary
 */
void EnforceBoundaryConditions(const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>>& fe_space,
                               lf::assemble::COOMatrix<double>& A,
                               Eigen::VectorXd& phi,
                               std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet
);

template<typename SCALAR>
class RotatingHumpSolution {
 public:
  RotatingHumpSolution() = default;
  ~RotatingHumpSolution() = default;
  RotatingHumpSolution(const RotatingHumpSolution &other) = default;
  RotatingHumpSolution(RotatingHumpSolution &&other) noexcept = default;
  RotatingHumpSolution &operator=(const RotatingHumpSolution &other) = default;
  RotatingHumpSolution &operator=(RotatingHumpSolution &&other) noexcept = default;

  explicit RotatingHumpSolution(
      const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>> &fe_space)
          : fe_space_(fe_space) {}

  Eigen::Vector<double, Eigen::Dynamic> ComputeSolution(
      std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double, 2, 1, 0>&)> velocity,
      std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double, 2, 1, 0>& )>
          boundary_condition,
      std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double, 2, 1, 0>&)> forcing_term
      ) const {
    //Wrap all functors into LehrFEM meshfunctions
    lf::mesh::utils::MeshFunctionGlobal mf_velocity{velocity};
    lf::mesh::utils::MeshFunctionGlobal mf_boundary_condition{
        boundary_condition};
    lf::mesh::utils::MeshFunctionGlobal mf_forcing_term{forcing_term};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    if(fe_space_->LocGlobMap().NumLocalDofs(*(mesh_p->EntityByIndex(0,0))) == 6) {
      // Quadratic FE space
      //todo
    }else if(fe_space_->LocGlobMap().NumLocalDofs(*(mesh_p->EntityByIndex(0,0))) == 3) {
      // Linear FE space
      // Assemble the first contribution matrix -- corresponding to  -Ru div(Rw)
      ecu_scheme::assemble::EdgeElementMassMatrixProvider<SCALAR, decltype(velocity)> edge_element_mass_matrix_provider(fe_space_, velocity);
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh, edge_element_mass_matrix_provider, A);
      // Assemble the second contribution matrix -- corresponding to grad(u \cdot w)
      ecu_scheme::assemble::EdgeElementGradMatrixProvider<SCALAR, decltype(velocity)> edge_element_grad_matrix_provider(fe_space_, velocity);
      lf::assemble::AssembleMatrixLocally(0, dofh, dofh, edge_element_grad_matrix_provider, A);
    }

    Eigen::VectorXd phi(N_dofs);
    phi.setZero();


    EnforceBoundaryConditions(fe_space_, A, phi, boundary_condition);
    

    Eigen::SparseMatrix<double> A_crs = A.makeSparse();
//    std::vector<Eigen::VectorXd> solution_vector;
//    solution_vector.push_back(lf::fe::NodalProjection(*fe_space_, mf_boundary_condition));
    Eigen::SparseLU<Eigen::SparseMatrix<double>> rot_hump_solver;
    rot_hump_solver.compute(A_crs);
    if(rot_hump_solver.info() != Eigen::Success){
      std::cerr << "LU Decomposition failed for Rotating Hump Experiment" << std::endl;
    }
    Eigen::VectorXd solution_vector = rot_hump_solver.solve(phi);
    if(rot_hump_solver.info() != Eigen::Success){
      std::cerr << "Solving failed for Rotating Hump Experiment" << std::endl;
    }

    return solution_vector;
  }

 private:
  std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space_;
};

}  // namespace ecu_scheme

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_EXPERIMENTS_ADVECTION_ONE_FORM_EXPERIMENTS_ROTATING_HUMP_ROTATING_HUMP_SOL_H_
