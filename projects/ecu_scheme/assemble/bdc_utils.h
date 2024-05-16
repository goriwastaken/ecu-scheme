
#ifndef LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_BDC_UTILS_H_
#define LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_BDC_UTILS_H_

#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>

#include <Eigen/Core>
#include <memory>

#include "lf/mesh/mesh.h"
#include "lf/uscalfe/uscalfe.h"

namespace ecu_scheme::assemble {

/**
 * @brief Enforce Dirichlet boundary conditions for the convection-diffusion
 * problem The boundary in question corresponds to the inflow boundary on the
 * unit rectangle where the velocity field is assumed to have the form
 * \f$\vec{v} = (c, c)^T\f$ for some constant \f$c\f$. Thus the inflow boundary
 * is \f$\Gamma = \Gamma_1 \cup \Gamma2 \f$ with \f$ \Gamma_1 = \{(x, y) \in
 * [0, 1]^2 \mid x = 0 \} \f$, and \f$ \Gamma_2 = \{(x, y) \in [0, 1]^2 \mid y =
 * 0 \} \f$.
 * @param fe_space underlying FE space
 * @param A Galerkin matrix
 * @param phi Right-hand side vector
 * @param dirichlet Dirichlet boundary condition function
 */
void EnforceBoundaryConditions(
    const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> &fe_space,
    lf::assemble::COOMatrix<double> &A, Eigen::VectorXd &phi,
    std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet);

/**
 * @brief Enforce Dirichlet boundary conditions for the convection-diffusion
 * problem The boundary corresponds to the inflow boundary on the unit rectangle
 * where we assume a rotational velocity field such that the inflow boundary has
 * the following form: \f$\Gamma = \Gamma_1 \cup \Gamma2 \f$ with \f$ \Gamma_1
 * = \{(x, y) \in [0, 1]^2 \mid x = 1 \} \f$, and \f$ \Gamma_2 = \{(x, y) \in
 * [0, 1]^2 \mid y = 0 \} \f$.
 * @param fe_space underlying FE space
 * @param A Galerkin matrix
 * @param phi Right-hand side vector
 * @param dirichlet Dirichlet boundary condition function
 */
void EnforceBoundaryConditionsOnRotInflow(
    const std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> &fe_space,
    lf::assemble::COOMatrix<double> &A, Eigen::VectorXd &phi,
    std::function<double(const Eigen::Matrix<double, 2, 1, 0> &)> dirichlet);

}  // namespace ecu_scheme::assemble

#endif  // LEHRFEMPP_PROJECTS_ECU_SCHEME_ASSEMBLE_BDC_UTILS_H_
