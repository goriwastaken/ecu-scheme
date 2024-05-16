#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_EXPERIMENTS_ADVECTION_ONE_FORM_EXPERIMENTS_ROTATING_HUMP_ROTATING_HUMP_SOL_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_EXPERIMENTS_ADVECTION_ONE_FORM_EXPERIMENTS_ROTATING_HUMP_ROTATING_HUMP_SOL_H_

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
 * @brief Computes the outward normals of a given triangular cell
 * @param entity entity (should be a triangle)
 * @return vector of outward normals
 */
Eigen::Matrix<double, 2, 3> computeOutwardNormals(
    const lf::mesh::Entity &entity);
/**
 * @brief Computes the tangential components of the edges of a triangular cell
 * @param entity reference to the entity (should be a triangle)
 * @return vector of tangential components in the order of how LehrFEM++ orders
 * the edges
 */
Eigen::Matrix<double, 2, 3> computeTangentialComponents(
    const lf::mesh::Entity &entity);
/**
 * @brief Enforce the Dirichlet boundary conditions at the inflow boundary which
 * is described by a rotational velocity field
 * @param fe_space underlying finite element space
 * @param A Galerkin system matrix
 * @param phi Right-hand side vector of the linear system
 * @param dirichlet Dirichlet function to be enforced at the inflow boundary
 */
void EnforceInflowBDCOneform(
    const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> &fe_space,
    lf::assemble::COOMatrix<double> &A, Eigen::VectorXd &phi,
    std::function<
        Eigen::Matrix<double, 2, 1>(const Eigen::Matrix<double, 2, 1, 0> &)>
        dirichlet);

/**
 * @brief Class for generating the Rotating Hump experiment setup and computing
 * the solution
 * @tparam SCALAR the scalar type of an underlying Lagrangian finite element
 * space
 */
template <typename SCALAR>
class RotatingHumpSolution {
 public:
  RotatingHumpSolution() = default;
  ~RotatingHumpSolution() = default;
  RotatingHumpSolution(const RotatingHumpSolution &other) = default;
  RotatingHumpSolution(RotatingHumpSolution &&other) noexcept = default;
  RotatingHumpSolution &operator=(const RotatingHumpSolution &other) = default;
  RotatingHumpSolution &operator=(RotatingHumpSolution &&other) noexcept =
      default;

  explicit RotatingHumpSolution(
      const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>>
          &fe_space)
      : fe_space_(fe_space) {}

  /**
   * @brief Computes the solution of the Rotating Hump experiment
   * @param velocity vector valued velocity field
   * @param boundary_condition vector valued Dirichlet boundary condition
   * @param forcing_term vector valued source term
   * @return solution vector
   */
  Eigen::Vector<double, Eigen::Dynamic> ComputeSolution(
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          velocity,
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          boundary_condition,
      std::function<Eigen::Matrix<double, 2, 1, 0>(
          const Eigen::Matrix<double, 2, 1, 0> &)>
          forcing_term) const {
    // Wrap all functors into LehrFEM meshfunctions
    lf::mesh::utils::MeshFunctionGlobal mf_velocity{velocity};
    lf::mesh::utils::MeshFunctionGlobal mf_boundary_condition{
        boundary_condition};
    lf::mesh::utils::MeshFunctionGlobal mf_forcing_term{forcing_term};

    auto mesh_p = fe_space_->Mesh();
    const lf::assemble::DofHandler &dofh{fe_space_->LocGlobMap()};
    const lf::uscalfe::size_type N_dofs(dofh.NumDofs());
    //    lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);
    // Our problem involves the edge element function space, namely the
    // one-form edge element basis functions We need to account for the
    // different DOFs  involved in the Nedelec FE space
    const bool predicate_fe_space = fe_space_->LocGlobMap().NumLocalDofs(
                                        *(mesh_p->EntityByIndex(0, 0))) == 3;
    const lf::assemble::DofHandler &dofh_edge =
        predicate_fe_space ? lf::assemble::UniformFEDofHandler(
                                 mesh_p, {{lf::base::RefEl::kSegment(), 1}})
                           : lf::assemble::UniformFEDofHandler(
                                 mesh_p, {{lf::base::RefEl::kSegment(), 2},
                                          {lf::base::RefEl::kTria(), 2}});
    const lf::assemble::size_type N_dofs_edge = dofh_edge.NumDofs();
    // Sizes of matrices are in the order: Nr. DOFs trial space, Nr. DOFs test
    // space

    lf::assemble::COOMatrix<double> A_edge_edge(N_dofs_edge, N_dofs_edge);
    lf::assemble::COOMatrix<double> A_debug(N_dofs_edge, N_dofs_edge);

    if (fe_space_->LocGlobMap().NumLocalDofs(*(mesh_p->EntityByIndex(0, 0))) ==
        6) {
      // Quadratic FE space
      // todo
    } else if (fe_space_->LocGlobMap().NumLocalDofs(
                   *(mesh_p->EntityByIndex(0, 0))) == 3) {
      // Linear FE space
      // Assemble the first contribution matrix -- corresponding to  -Ru div(Rw)
      ecu_scheme::assemble::EdgeElementMassMatrixProvider<SCALAR,
                                                          decltype(velocity)>
          edge_element_mass_matrix_provider(fe_space_, velocity);
      lf::assemble::AssembleMatrixLocally(
          0, dofh_edge, dofh_edge, edge_element_mass_matrix_provider, A_debug);
      // Assemble the second contribution matrix -- corresponding to grad(u
      // \cdot w)
      ecu_scheme::assemble::EdgeElementGradMatrixProvider<SCALAR,
                                                          decltype(velocity)>
          edge_element_grad_matrix_provider(fe_space_, velocity);
      lf::assemble::AssembleMatrixLocally(
          0, dofh_edge, dofh_edge, edge_element_grad_matrix_provider, A_debug);
    }

    Eigen::VectorXd phi(N_dofs_edge);
    phi.setZero();

    EnforceInflowBDCOneform(fe_space_, A_debug, phi, boundary_condition);

    Eigen::SparseMatrix<double> A_crs = A_debug.makeSparse();
    //    std::vector<Eigen::VectorXd> solution_vector;
    //    solution_vector.push_back(lf::fe::NodalProjection(*fe_space_,
    //    mf_boundary_condition));
    Eigen::SparseLU<Eigen::SparseMatrix<double>> rot_hump_solver;
    rot_hump_solver.compute(A_crs);
    if (rot_hump_solver.info() != Eigen::Success) {
      std::cerr << "LU Decomposition failed for Rotating Hump Experiment"
                << std::endl;
    }
    Eigen::VectorXd solution_vector = rot_hump_solver.solve(phi);
    if (rot_hump_solver.info() != Eigen::Success) {
      std::cerr << "Solving failed for Rotating Hump Experiment" << std::endl;
    }

    return solution_vector;
  }

 private:
  std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space_;
};

}  // namespace ecu_scheme::experiments

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_EXPERIMENTS_ADVECTION_ONE_FORM_EXPERIMENTS_ROTATING_HUMP_ROTATING_HUMP_SOL_H_
