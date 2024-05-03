#include <cmath>

#include "rotating_hump_sol.h"
// Other libraries headers
#include "lf/uscalfe/uscalfe.h"
#include "mesh.h"
#include "post_processing.h"
// remove assemble after moving convergence report in results_processing
#include <Eigen/Core>
#include <iostream>

#include "assemble.h"

/**
 * @brief Rotating Hump Problem for 2-dimensional pure transport problem of
 * differential 1-forms
 */
int main(int argc, char* argv[]) {
  if (argc != 2 && argc != 1 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << " refinement_levels eps " << std::endl;
    return -1;
  }
  // We inspect a pure transport problem
  double eps_for_refinement = 0;
  // Number of refinement levels
  unsigned int refinement_levels = 3;

  // Adjust the number of refinement levels and the diffusion coefficient if the
  // user specified them
  if (argc == 2) {
    refinement_levels = std::stoi(argv[1]);
  }else if(argc == 3){
    refinement_levels = std::stoi(argv[1]);
    eps_for_refinement = 0; //pure transport problem
  }
  std::cout << "Refinement levels: " << refinement_levels << '\n';
  std::cout << "Epsilon for refinement: " << eps_for_refinement << '\n';

  // setup is \Omega = [-1, 1]^2
  // Lipschitz continuous velocity field
  // see Thesis chapter Numerical Experiments section 5.2

  ecu_scheme::mesh::BasicMeshBuilder builder;
  builder.SetNumCellsX(49);
  builder.SetNumCellsY(49);
  std::shared_ptr<lf::mesh::Mesh> mesh_p = builder.Build(0.0, 0.0, 1.0, 1.0);



  // Define rotational velocity field
  const auto kVelocityField = [](const Eigen::Vector2d& xh) {
    return (Eigen::Vector2d() << -xh(1), xh(0)).finished();
  };
  const auto kBoundaryConditionBump = [](const Eigen::Vector2d& xh) {
    const double x_norm = xh.norm();
    if (x_norm <= 1.0) {
      const double phi = std::pow(std::sin(M_PI * x_norm), 2);
      return (Eigen::Vector2d() << phi, phi).finished();
    }
    return (Eigen::Vector2d() << 0.0, 0.0).finished();
  };
  const auto kForcingTerm = [](const Eigen::Vector2d& xh) {
    return (Eigen::Vector2d() << 0.0, 0.0).finished();
  };
  const auto kPhiRotational = [](const Eigen::Vector3d& xh) {
    // xh has the form: (\theta, x_1, x_2)
    Eigen::MatrixXd rotation_matrix;
    rotation_matrix << std::cos(xh(0)), -std::sin(xh(0)), std::sin(xh(0)),
        std::cos(xh(0));
    Eigen::Vector2d result =
        rotation_matrix * (Eigen::Vector2d() << xh(1), xh(2)).finished();
    return result;
  };
  const auto kRoot = [](const Eigen::Vector2d& xh) {
    const double radius_norm = xh.norm();
    return (Eigen::Vector2d() << radius_norm, 0.0).finished();
  };
  const auto kAngle = [](const Eigen::Vector2d& xh) {
    return std::atan2(xh(1), xh(0));
  };
  const auto JacobianInverseTransposePhi = [kAngle](const Eigen::Vector2d& xh) {
    Eigen::Matrix2d result;
    result.setZero();
    result << std::cos(kAngle(xh)), -std::sin(kAngle(xh)), std::sin(kAngle(xh)),
        std::cos(kAngle(xh));
    return result;
  };
  const auto kExactSolution =
      [JacobianInverseTransposePhi,
       kBoundaryConditionBump](const Eigen::Vector2d& xh) {
        Eigen::Vector2d result = (Eigen::Vector2d() << 0.0, 0.0).finished();
        result = JacobianInverseTransposePhi(xh) * kBoundaryConditionBump(xh);
        return result;
      };
  // Wrap exact solution as a Mesh Function
  lf::mesh::utils::MeshFunctionGlobal mf_exact_sol{kExactSolution};

  // Regular refinement tests
  std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(
          mesh_p, refinement_levels);
  lf::refinement::MeshHierarchy& multi_mesh{*mesh_hierarchy};
  multi_mesh.PrintInfo(std::cout);

  ecu_scheme::post_processing::ExperimentSolutionWrapper<double>
      solution_collection_wrapper_linear{
          refinement_levels, eps_for_refinement,
          Eigen::VectorXd::Zero(refinement_levels), mesh_hierarchy,
          std::vector<Eigen::VectorXd>{}};

  auto L = multi_mesh.NumLevels();
  for (int l = 0; l < L; ++l) {
    // Compute the FE solution for every level
    auto mesh_l = multi_mesh.getMesh(l);

    auto fe_space_l_linear =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_l);

    // Compute the solution for the linear FE space
    ecu_scheme::experiments::RotatingHumpSolution<double> experiment_l_linear(
        fe_space_l_linear);
    const int kNrTimeStepsLinear = 10;
    Eigen::VectorXd solution_vector_l_linear =
        experiment_l_linear.ComputeSolution(
            kVelocityField, kBoundaryConditionBump, kForcingTerm);

    solution_collection_wrapper_linear.final_time_solutions.push_back(
        solution_vector_l_linear);
  }

  // Realize convergence result for uniform refinement
  ecu_scheme::post_processing::convergence_report_oneform<double, decltype(mf_exact_sol)>(
      solution_collection_wrapper_linear, mf_exact_sol,
      ecu_scheme::post_processing::concat(
          "rot_hump_linear","_",refinement_levels,"_",eps_for_refinement
          ), true
      );

  //debug L2 error computation
//  for (int l = 0; l < L; ++l) {
//    auto mesh_l = multi_mesh.getMesh(l);
//    auto fe_space_l_linear =
//        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_l);
//
//    Eigen::VectorXd final_time_sol_vector =
//        solution_collection_wrapper_linear.final_time_solutions.at(l);
//    ecu_scheme::assemble::MeshFunctionOneForm<double> mf_one_form(
//        final_time_sol_vector, mesh_l);
//
//
//    //    auto mf_diff = lf::uscalfe::operator-(mf_one_form, mf_exact_sol);
//    auto mf_diff = mf_one_form - mf_exact_sol;
//
//    // prepare extra info for L2norm function
//    auto square_vector =
//        [](Eigen::Matrix<double, Eigen::Dynamic, 1> a) -> double {
//      return a.squaredNorm();
//    };
//    lf::quad::QuadRule quad_rule =
//        lf::quad::make_QuadRule(lf::base::RefEl::kTria(), 10);
//
//    // get L2_error
//    const std::pair<double, lf::mesh::utils::CodimMeshDataSet<double>>
//        L2_error_bundle = ecu_scheme::post_processing::L2norm(
//            mesh_l, mf_diff, square_vector, quad_rule);
//    std::cout << "Ndofs: " << fe_space_l_linear->LocGlobMap().NumDofs()
//              << " L2-error: " << std::get<0>(L2_error_bundle) << "\n";
//  }

  return 0;
}
