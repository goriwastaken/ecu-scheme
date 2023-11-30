/**
 * @file upwindquadrature_main.cc
 * @brief NPDE homework template main
 * @author Philippe Peter
 * @date June 2020
 * @copyright Developed at SAM, ETH Zurich
 */
#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/mesh.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <cmath>
#include <memory>

#include "test/convection_emp.h"
#include "upwindquadrature.h"

int main() {
  // PARAMETERS
  // mesh specification (number of cells in both sides of the tensor-product
  // triangular mesh)
  int M = 49;

  // coefficient functions:
  // Dirichlet functor
  const auto g = [](const Eigen::Vector2d &x) {
    return x(1) == 0 ? 0.5 - std::abs(x(0) - 0.5) : 0.0;
  };
  lf::mesh::utils::MeshFunctionGlobal mf_g{g};

  // velocity field
  const auto v = [](const Eigen::Vector2d &x) {
    return (Eigen::Vector2d() << -x(1), x(0)).finished();
  };

  // diffusion coefficient
  const double eps = 1e-4;
  lf::mesh::utils::MeshFunctionConstant mf_eps{eps};

  // MESH CONSTRUCTION
  // construct a triangular tensor product mesh on the unit square
  std::unique_ptr<lf::mesh::MeshFactory> mesh_factory_ptr =
      std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::mesh::utils::TPTriagMeshBuilder builder(std::move(mesh_factory_ptr));
  builder.setBottomLeftCorner(Eigen::Vector2d{0.0, 0.0})
      .setTopRightCorner(Eigen::Vector2d{1.0, 1.0})
      .setNumXCells(M)
      .setNumYCells(M);
  std::shared_ptr<lf::mesh::Mesh> mesh_p = builder.Build();

  // DOF HANDLER & FINITE ELEMENT SPACE
  // Construct dofhanlder for linear finite elements on the mesh.
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);
  const lf::assemble::DofHandler &dofh{fe_space->LocGlobMap()};

  std::cout << dofh.NumDofs() << " dofs of Linear FE" << '\n';
  //output global dofs of some entity
  std::cout << dofh.GlobalDofIndices(*mesh_p->EntityByIndex(0,0)).size() <<" globdofs LINEAR of first entity" << '\n';


  // PREPARING DATA TO IMPOSE DIRICHLET CONDITIONS
  // Obtain specification for shape functions on edges
  const lf::fe::ScalarReferenceFiniteElement<double> *rsf_edge_p =
      fe_space->ShapeFunctionLayout(lf::base::RefEl::kSegment());

  // Create a dataset of boolean flags indicating edges on the boundary of the
  // mesh
  auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};

  // Fetch flags and values for degrees of freedom located on Dirichlet
  // boundary.
  auto ess_bdc_flags_values{
      lf::fe::InitEssentialConditionFromFunction(*fe_space, bd_flags, mf_g)};

  //============================================================================
  // SOLVE LAPLACIAN WITH NON-HOMOGENEOUS DIRICHLET BC (STANDARD: UNSTABLE)
  //============================================================================
  // Matrix in triplet format holding Galerkin matrix, zero initially.
  lf::assemble::COOMatrix<double> A(dofh.NumDofs(), dofh.NumDofs());

  // ASSEMBLE GALERKIN MATRIX
  // First the part corresponding to the laplacian
  lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider(
      fe_space, mf_eps, lf::mesh::utils::MeshFunctionConstant(0.0));
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider, A);

  // Next part corresponding to the convection term:
  ConvectionDiffusion::ConvectionElementMatrixProvider convection_provider(v);
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, convection_provider, A);

  // RIGHT-HAND SIDE VECTOR
  Eigen::VectorXd phi(dofh.NumDofs());
  phi.setZero();

  // IMPOSE DIRICHLET CONDITIONS:
  // Eliminate Dirichlet dofs from linear system
  lf::assemble::FixFlaggedSolutionComponents<double>(
      [&ess_bdc_flags_values](lf::uscalfe::glb_idx_t gdof_idx) {
        return ess_bdc_flags_values[gdof_idx];
      },
      A, phi);

  // SOLVE LINEAR SYSTEM
  Eigen::SparseMatrix A_crs = A.makeSparse();
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_crs);
  Eigen::VectorXd sol_vec = solver.solve(phi);

  // OUTPUT RESULTS TO VTK FILe
  // construct mesh function representing finite element solution
  lf::fe::MeshFunctionFE mf_sol(fe_space, sol_vec);
  // construct vtk writer
  lf::io::VtkWriter vtk_writer(
      mesh_p, CURRENT_BINARY_DIR "/upwind_quadrature_solution_unstable.vtk");
  // output data
  vtk_writer.WritePointData("upwind_quadrature_solution_unstable", mf_sol);

//============================================================================
// SOLVE LAPLACIAN WITH NON-HOMOGENEOUS DIRICHLET BC (UPWIND: STABLE)
//============================================================================
/* SAM_LISTING_BEGIN_7 */
  //====================
  lf::assemble::COOMatrix<double> A_upwind(dofh.NumDofs(),dofh.NumDofs());

  //diffusive part
  lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider_upwind(
      fe_space, mf_eps, lf::mesh::utils::MeshFunctionConstant(0.0)
      );
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, laplacian_provider_upwind, A_upwind);
  //convective part
  UpwindQuadrature::UpwindConvectionElementMatrixProvider upwind_convection_element_matrix_provider(v,
                                                                                                    UpwindQuadrature::initializeMasses(mesh_p));
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, upwind_convection_element_matrix_provider, A_upwind);

  //RHS vector
  Eigen::VectorXd phi_upwind(dofh.NumDofs());
  phi_upwind.setZero();

  //IMPOSE DIRICHLET CONDITIONS:
  lf::assemble::FixFlaggedSolutionComponents<double>(
      [&ess_bdc_flags_values](lf::uscalfe::glb_idx_t gdof_idx) {
        return ess_bdc_flags_values[gdof_idx];
      },
      A_upwind, phi_upwind);

  // SOLVE LINEAR SYSTEM
  Eigen::SparseMatrix A_crs_upwind = A_upwind.makeSparse();
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_stable;
  solver_stable.compute(A_crs_upwind);
  Eigen::VectorXd sol_vec_stable = solver_stable.solve(phi_upwind);

  // OUTPUT RESULTS TO VTK FILe
  // construct mesh function representing finite element solution
  lf::fe::MeshFunctionFE mf_sol_stable(fe_space, sol_vec_stable);
  // construct vtk writer
  lf::io::VtkWriter vtk_writer_stable(
      mesh_p, CURRENT_BINARY_DIR "/upwind_quadrature_solution_stable.vtk");
  // output data
  vtk_writer_stable.WritePointData("upwind_quadrature_solution_stable", mf_sol_stable);
  //====================

  //=====================================================================================
  // LAPLACIAN WITH NON-HOMOGENEOUS DIRICHLET BC FOR QUADRATIC FE (UPWIND: STABLE)
  //=====================================================================================

  // DOF HANDLER & FINITE ELEMENT SPACE
  // Construct dofhanlder for quadratic finite elements on the mesh.
  auto fe_space_quad =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
  const lf::assemble::DofHandler &dofh_q{fe_space_quad->LocGlobMap()};

  std::cout << dofh_q.NumDofs() << " dofs of Quadratic FE" << '\n';
  //output global dofs of some entity
  std::cout << dofh_q.GlobalDofIndices(*mesh_p->EntityByIndex(0,0)).size() <<" globdofs of first entity" << '\n';
  std::cout << (mesh_p->Entities(0).size()) << " nr of mesh cells" << '\n';

  // Fetch flags and values for degrees of freedom located on Dirichlet
  // boundary.
  //auto bd_edge_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 2)};
  auto ess_bdc_flags_values_quad{
      lf::fe::InitEssentialConditionFromFunction(*fe_space_quad, bd_flags, mf_g)};

  // SOLVE LAPLACIAN WITH NON-HOMOGENEOUS DIRICHLET BC FOR QUADRATIC FE (UPWIND: STABLE)
  lf::assemble::COOMatrix<double> A_upwind_quad(dofh_q.NumDofs(), dofh_q.NumDofs());

  // diffusive part
  lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider_upwind_quad(
      fe_space_quad, mf_eps, lf::mesh::utils::MeshFunctionConstant(0.0)
      );
  lf::assemble::AssembleMatrixLocally(0, dofh_q, dofh_q, laplacian_provider_upwind_quad, A_upwind_quad);

  // convective part
  UpwindQuadratureQuadratic::UpwindQuadratureQuadraticElementMatrixProvider upwind_quadrature_quadratic_element_matrix_provider(
      v, UpwindQuadratureQuadratic::initializeMassesQuadratic(mesh_p), UpwindQuadratureQuadratic::initializeMassesQuadraticEdges(mesh_p)
      );
  std::cout << A_upwind_quad.cols() << ' ' << A_upwind_quad.rows() <<" dims of A matr" <<'\n';
  std::cout<< dofh_q.NumDofs() << " dofs" << '\n';
  //const lf::mesh::Entity &element_0 = *(mesh_p->EntityByIndex(0, 0));

//  Eigen::MatrixXd reference_eval = laplacian_provider_upwind_quad.Eval(element_0);
//  //Eigen::MatrixXd upwind_eval = upwind_quadrature_quadratic_element_matrix_provider.Eval(element_0);
//  Eigen::MatrixXd upwind_eval = upwind_quadrature_quadratic_element_matrix_provider.Eval(element_0);
//  std::cout << "compare rows evals: " << reference_eval.rows() << ' '<<upwind_eval.rows() << '\n';
////  static_assert(reference_eval.rows() == upwind_eval.rows(), "rows do not match");
////  static_assert(reference_eval.cols() == upwind_eval.cols(), "cols do not match");
//  std::cout << "compare cols evals: " << reference_eval.cols() << ' ' <<upwind_eval.cols()<< '\n';
  lf::assemble::AssembleMatrixLocally(0, dofh_q, dofh_q,
                                      upwind_quadrature_quadratic_element_matrix_provider,
                                      A_upwind_quad);
//  for(const lf::mesh::Entity *e: mesh_p->Entities(0)){
//    Eigen::MatrixXd upwind_eval = upwind_quadrature_quadratic_element_matrix_provider.Eval(*e);
//    std::cout << "compare rows evals: " << upwind_eval.rows() << '\n';
//  }
  // RHS vector
  Eigen::VectorXd phi_quad(dofh_q.NumDofs());
  phi_quad.setZero();

  // IMPOSE DIRICHLET CONDITIONS:
  //UpwindQuadratureQuadratic::enforce_boundary_conditions(fe_space_quad, A_upwind_quad, phi_quad);
  lf::assemble::FixFlaggedSolutionComponents<double>(
      [&ess_bdc_flags_values_quad](lf::uscalfe::glb_idx_t gdof_idx) {
        return ess_bdc_flags_values_quad[gdof_idx];
      },
      A_upwind_quad, phi_quad);
//  lf::assemble::FixFlaggedSolutionComponents<double>(
//      [&ess_bdc_flags_values_quad](lf::uscalfe::glb_idx_t gdof_idx) {
//        return ess_bdc_flags_values_quad[gdof_idx];
//      },
//      A_upwind_quad, phi_quad);


  // SOLVE LINEAR SYSTEM
  Eigen::SparseMatrix A_crs_quad = A_upwind_quad.makeSparse();
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_quad;
  solver_quad.compute(A_crs_quad);
  //assert(solver_quad.info()!=Eigen::Success);
  Eigen::VectorXd sol_vec_quad = solver_quad.solve(phi_quad);

  // todo OUTPUT RESULTS TO VTK FILE
  // construct mesh function representing finite element solution
  lf::fe::MeshFunctionFE mf_sol_quad(fe_space_quad, sol_vec_quad);
  // construct vtk writer
  lf::io::VtkWriter vtk_writer_quad(
      mesh_p, CURRENT_BINARY_DIR "/upwind_quadrature_solution_quadratic.vtk");
  // output data
  vtk_writer_quad.WritePointData("upwind_quadrature_solution_quadratic", mf_sol_quad);
  //====================

  //=====================================================================================
  // HOLGER HEUMANN EXPERIMENT FOR QUADRATIC FE
  //=====================================================================================

  auto fe_space_heumann =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
//  auto fe_space_heumann = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);
  const lf::assemble::DofHandler &dofh_heumann{fe_space_heumann->LocGlobMap()};

  // Heumann experiment diffusion coefficient
  const double eps_h = 1e-10;
  lf::mesh::utils::MeshFunctionConstant mf_eps_h{-eps_h};

  // velocity field Heumann exp
  const auto v_heumann = [](const Eigen::Vector2d &x) {
    return (Eigen::Vector2d() << 2.0, 3.0).finished();
  };

  // Test function f manufactured sol
  const auto f_manuf = [eps_h](const Eigen::Vector2d &x){
    return 2.0 * (-eps_h*x(0) + 3.0*x(0)*x(1) + x(1)*x(1) + (eps_h - 3.0*x(1))*exp(2.0*(-1.0 + x(0))/eps_h)
                  - exp(3.0*(-1.0+x(1))/eps_h));
  };
  lf::mesh::utils::MeshFunctionGlobal mf_f_manuf{f_manuf};

  // Dirichlet functor manufactured sol
  const auto g_manuf = [eps_h](const Eigen::Vector2d &x) {
    if(x(0) < 1e-5){
      return -(x(1)*x(1))*exp(-2.0/eps_h) + exp(-2.0/eps_h + 3.0*(-1.0+x(1))/eps_h);
    }else if(x(0) > 1.0 - 1e-5) {
      return 0.0;
    }else if(x(1) < 1e-5){
      return -x(0)*exp(-3.0/eps_h) + exp(2.0*(-1.0+x(0))/eps_h - 3.0/eps_h);
    }else if(x(1) > 1.0 - 1e-5){
      return 0.0;
    }else return 0.0;
  };
  lf::mesh::utils::MeshFunctionGlobal mf_g_manuf{g_manuf};

  auto ess_bdc_flags_values_heumann{
      lf::fe::InitEssentialConditionFromFunction(*fe_space_heumann, bd_flags, mf_g_manuf)};

  lf::assemble::COOMatrix<double> A_heumann(dofh_heumann.NumDofs(), dofh_heumann.NumDofs());

  lf::uscalfe::ReactionDiffusionElementMatrixProvider laplacian_provider_heumann(
      fe_space_heumann, mf_eps_h, lf::mesh::utils::MeshFunctionConstant(0.0)
      );

  lf::assemble::AssembleMatrixLocally(0, dofh_heumann, dofh_heumann, laplacian_provider_heumann, A_heumann);

  UpwindQuadratureQuadratic::UpwindQuadratureQuadraticElementMatrixProvider upwind_quadrature_quadratic_element_matrix_provider_heumann(
      v_heumann, UpwindQuadratureQuadratic::initializeMassesQuadratic(mesh_p), UpwindQuadratureQuadratic::initializeMassesQuadraticEdges(mesh_p)
      );
//  UpwindQuadrature::UpwindConvectionElementMatrixProvider upwind_quadrature_quadratic_element_matrix_provider_heumann(
//      v_heumann, UpwindQuadrature::initializeMasses(mesh_p)
//      );

  lf::assemble::AssembleMatrixLocally(0, dofh_heumann, dofh_heumann,
                                        upwind_quadrature_quadratic_element_matrix_provider_heumann,
                                        A_heumann);

  Eigen::VectorXd phi_heumann(dofh_heumann.NumDofs());
  lf::fe::ScalarLoadElementVectorProvider<double, decltype(mf_f_manuf)> load_provider_phi(fe_space_heumann, mf_f_manuf);

  lf::assemble::AssembleVectorLocally(0, dofh_heumann, load_provider_phi, phi_heumann);

  //UpwindQuadratureQuadratic::enforce_boundary_conditions(fe_space_heumann, A_heumann, phi_heumann);
    lf::assemble::FixFlaggedSolutionComponents<double>(
        [&ess_bdc_flags_values_heumann](lf::uscalfe::glb_idx_t gdof_idx) {
            return ess_bdc_flags_values_heumann[gdof_idx];
        },
        A_heumann, phi_heumann);

  Eigen::SparseMatrix A_crs_heumann = A_heumann.makeSparse();
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_heumann;
  solver_heumann.compute(A_crs_heumann);
  Eigen::VectorXd sol_vec_heumann = solver_heumann.solve(phi_heumann);

  // OUTPUT TO VTK FILE
  lf::fe::MeshFunctionFE mf_sol_heumann(fe_space_heumann, sol_vec_heumann);
  lf::io::VtkWriter vtk_writer_heumann(
        mesh_p, CURRENT_BINARY_DIR "/upwind_quadrature_solution_heumann.vtk");
  vtk_writer_heumann.WritePointData("upwind_quadrature_solution_heumann", mf_sol_heumann);

  //=========================================

  /* SAM_LISTING_END_7 */
  return 0;
}
