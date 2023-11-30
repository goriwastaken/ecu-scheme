// Project header file include
// C system headers
// C++ standard library headers
// Other libraries headers

#ifndef THESIS_EXPERIMENTS_CONST_VELO_CONST_VELO_SOLUTION_H_
#define THESIS_EXPERIMENTS_CONST_VELO_CONST_VELO_SOLUTION_H_

#include <functional>
#include <utility>

#include "lf/assemble/assemble.h"
#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"
#include "lf/fe/fe.h"

#include "Eigen/Core"

#include "assemble.h"

namespace ecu_scheme::experiments {

template <typename SCALAR>
class ConstVeloSolution {
 public:
  ConstVeloSolution() = default;
  ~ConstVeloSolution() = default;

  Eigen::VectorXd step(const Eigen::VectorXd& u0_vector, double tau, std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double,2,1,0>&)> velocity);

  explicit ConstVeloSolution(std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space)
      : fe_space_(std::move(fe_space)) {}

  Eigen::Vector<double, Eigen::Dynamic> ComputeSolution(std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> initialCondition,
                                                        std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double,2,1,0>&)> velocity
                                                        ) const{
    // Wrap all functors into LehrFEM MeshFunctions
    lf::mesh::utils::MeshFunctionGlobal<decltype(initialCondition)> mf_initial_condition{initialCondition};
    lf::mesh::utils::MeshFunctionGlobal<decltype(velocity)> mf_velocity{velocity};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

    //ecu_scheme::assemble::ConvectionUpwindMatrixProvider convection_upwind_provider(fe_space_, mf_velocity, ecu_scheme::assemble::initializeMassesQuadratic(mesh_p), ecu_scheme::assemble::initializeMassesQuadraticEdges(mesh_p));
    //todo fix stepping function



  }

 private:
  std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space_;
  Eigen::SparseMatrix<double> convection_matrix_sparse_;
};

template<typename SCALAR>
Eigen::VectorXd ConstVeloSolution<SCALAR>::step(const Eigen::VectorXd &u0_vector,
                                                double tau,
                                                std::function<Eigen::Matrix<double, 2, 1, 0>(const Eigen::Matrix<double,2,1,0>&)> velocity) {
  Eigen::VectorXd result = Eigen::VectorXd::Zero(u0_vector.size());

  // explicit midpoint rule computes solution to u'(t) = f(u(t)) with u(0) = u0
  // y^* = y_n + tau/2 * f(y_n)
  // y_{n+1} = y_n + tau * f(y^*)

  // implement this for the pde d/dt u = -v * grad u
  // get element matrix
  //ecu_scheme::assemble::ConvectionUpwindMatrixProvider elem_mat_provider(fe_space_, mf_velocity)
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;


  return result;
}

} // experiments

#endif //THESIS_EXPERIMENTS_CONST_VELO_CONST_VELO_SOLUTION_H_
