// Project header file include
#include "concentric_stream_solution.h"
// C system headers
// C++ standard library headers
#include <cmath>
// Other libraries headers
#include <Eigen/Core>
#include <Eigen/Dense>

#include "lf/uscalfe/uscalfe.h"
#include "mesh.h"
#include "post_processing.h"

/**
 * @brief Concentric Streamlines Problem for 2-dimensional pure transport
 * problem from section 5.1.2 of the thesis
 */
int main(int argc, char* argv[]) {
  if (argc != 3 && argc != 1) {
    std::cerr << "Usage: " << argv[0] << " refinement_levels eps " << std::endl;
    return -1;
  }
  // Concentric circles diffusion coefficient
  double eps_for_refinement = 0;
  // Number of refinement levels
  unsigned int refinement_levels = 7;
  // Adjust the number of refinement levels and the diffusion coefficient if the
  // user specified them
  if (argc == 3) {
    refinement_levels = std::stoi(argv[1]);
    eps_for_refinement = 0.0;
    //    if (eps_for_refinement > 0.0) {
    //      std::cerr << "This experiment concerns the pure transport problem"
    //                << std::endl;
    //    }
  }
  std::cout << "Refinement levels: " << refinement_levels << '\n';
  std::cout << "Epsilon for refinement: " << eps_for_refinement << '\n';

  ecu_scheme::mesh::BasicMeshBuilder builder;
  builder.SetNumCellsX(2);
  builder.SetNumCellsY(2);
  std::shared_ptr<lf::mesh::Mesh> mesh_p = builder.Build();

  auto fe_space_quad =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
  auto fe_space_linear =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);

  // Concentric circles diffusion coefficient
  const double kEps = eps_for_refinement;

  // Concentric circle streamline velocity field
  const auto kVelocity = [](const Eigen::Vector2d& x) {
    return (Eigen::Vector2d() << -x(1), x(0)).finished();
  };

  // 0 <= xi <= 1
  const auto kPhiRotational = [](const double& xi) {
    return std::sin(M_PI * xi) * std::sin(M_PI * xi);
  };

  const auto kDirichletFunctor = [kPhiRotational](const Eigen::Vector2d& x) {
    auto xnorm = x.norm();
    if (xnorm > 1.0) {
      return 0.0;
    }
    return kPhiRotational(xnorm);
  };

  // Test function for the concentric circle streamlines problem
  const auto kTestFunctor = [](const Eigen::Vector2d& x) { return 0.0; };

  // Exact solution for the concentric circle streamlines problem
  const auto kExact = [kPhiRotational](const Eigen::Vector2d& x) {
    auto norm = x.norm();
    if (norm > 1.0) {
      return 0.0;
    }
    return kPhiRotational(norm);
  };

  // Wrap exact solution into meshfunction
  lf::mesh::utils::MeshFunctionGlobal mf_u_exact{kExact};
  // Wrap all functors into LehrFEM MeshFunctions
  lf::mesh::utils::MeshFunctionConstant mf_eps{kEps};
  lf::mesh::utils::MeshFunctionGlobal mf_velocity{kVelocity};
  lf::mesh::utils::MeshFunctionGlobal mf_g_dirichlet{kDirichletFunctor};

  lf::mesh::utils::MeshFunctionGlobal mf_test_exact_plot{kExact};
  Eigen::VectorXd exact_test_vector =
      lf::fe::NodalProjection(*fe_space_linear, mf_test_exact_plot);
  lf::fe::MeshFunctionFE mf_test_exact_linear(fe_space_linear,
                                              exact_test_vector);
  ecu_scheme::post_processing::output_meshfunction_paraview<
      double, decltype(mf_test_exact_linear)>(
      fe_space_linear, mf_test_exact_linear,
      "concentric_stream_solution_test_exact");

  Eigen::VectorXd exact_test_vector_quad =
      lf::fe::NodalProjection(*fe_space_quad, mf_test_exact_plot);
  ecu_scheme::post_processing::output_results<double>(
      fe_space_quad, exact_test_vector_quad,
      "concentric_stream_solution_test_exact_quad");

  //  lf::mesh::utils::MeshFunctionGlobal mf_test_exact{test_exact};
  //  Eigen::VectorXd exact_test_vector =
  //  lf::fe::NodalProjection(*fe_space_linear, mf_u_exact);
  //
  //  ecu_scheme::post_processing::output_results<double>(
  //      fe_space_linear, exact_test_vector, "concentric_stream_solution");

  // Compute the L2 error norms with regular refinement
  std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(
          mesh_p, refinement_levels);
  lf::refinement::MeshHierarchy& multi_mesh{*mesh_hierarchy_p};
  multi_mesh.PrintInfo(std::cout);

  // Initialize solution wrapper for quadratic FE
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double>
      solution_collection_wrapper_quad{refinement_levels, eps_for_refinement,
                                       Eigen::VectorXd::Zero(refinement_levels),
                                       mesh_hierarchy_p,
                                       std::vector<Eigen::VectorXd>{}};
  // Initialize solution wrapper for linear FE
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double>
      solution_collection_wrapper_linear{
          refinement_levels, eps_for_refinement,
          Eigen::VectorXd::Zero(refinement_levels), mesh_hierarchy_p,
          std::vector<Eigen::VectorXd>{}};

  // Solution wrappers for SUPG comparison
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double>
      solution_collection_wrapper_supg_linear{
          refinement_levels, eps_for_refinement,
          Eigen::VectorXd::Zero(refinement_levels), mesh_hierarchy_p,
          std::vector<Eigen::VectorXd>{}};

  ecu_scheme::post_processing::ExperimentSolutionWrapper<double>
      solution_collection_wrapper_supg_quad{
          refinement_levels, eps_for_refinement,
          Eigen::VectorXd::Zero(refinement_levels), mesh_hierarchy_p,
          std::vector<Eigen::VectorXd>{}};

  // debug with other methods
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double>
      solution_collection_wrapper_quad_stable{
          refinement_levels, eps_for_refinement,
          Eigen::VectorXd::Zero(refinement_levels), mesh_hierarchy_p,
          std::vector<Eigen::VectorXd>{}};

  ecu_scheme::post_processing::ExperimentSolutionWrapper<double>
      solution_collection_wrapper_quad_fifteen_upwind{
          refinement_levels, eps_for_refinement,
          Eigen::VectorXd::Zero(refinement_levels), mesh_hierarchy_p,
          std::vector<Eigen::VectorXd>{}};

  // get number of levels
  auto L = multi_mesh.NumLevels();
  LF_ASSERT_MSG(L == refinement_levels + 1, "Number of levels is not correct");
  for (int l = 0; l < L; ++l) {
    auto mesh_l = multi_mesh.getMesh(l);
    // Quadratic case
    auto fe_space_l =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_l);
    ecu_scheme::experiments::ConcentricStreamSolution<double> experiment_l(
        fe_space_l);

    // midpoint upwind scheme
    //    Eigen::VectorXd solution_vector_l =
    //    experiment_l.ComputeSolutionMultipleMethods(kEps, kVelocity,
    //    kDirichletFunctor, kTestFunctor, "UPWIND");
    //    solution_collection_wrapper_quad.final_time_solutions.push_back(solution_vector_l);
    // 7-point scheme from HH08
    Eigen::VectorXd solution_vector_l_stable_upwind =
        experiment_l.ComputeSolutionMultipleMethods(
            kEps, kVelocity, kDirichletFunctor, kTestFunctor, "STABLE_UPWIND");
    solution_collection_wrapper_quad_stable.final_time_solutions.push_back(
        solution_vector_l_stable_upwind);
    // 15 point upwind scheme
    Eigen::VectorXd solution_vector_l_fifteen_pt_upwind =
        experiment_l.ComputeSolutionMultipleMethods(
            kEps, kVelocity, kDirichletFunctor, kTestFunctor, "15P_UPWIND");
    solution_collection_wrapper_quad_fifteen_upwind.final_time_solutions
        .push_back(solution_vector_l_fifteen_pt_upwind);

    // SUPG method - quadratic FE space
    const auto kTempEps = [kEps](const Eigen::Vector2d& x) { return kEps; };
    Eigen::VectorXd solution_vector_supg_quad =
        ecu_scheme::assemble::SolveCDBVPSupgQuad<
            decltype(kTempEps), decltype(kVelocity), decltype(kTestFunctor),
            decltype(kDirichletFunctor)>(fe_space_l, kTempEps, kVelocity,
                                         kTestFunctor, kDirichletFunctor, true);
    solution_collection_wrapper_supg_quad.final_time_solutions.push_back(
        solution_vector_supg_quad);

    // Linear case
    auto fe_space_l_linear =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_l);
    //    ecu_scheme::experiments::ConcentricStreamSolution<double>
    //    experiment_l_linear(fe_space_l_linear); Eigen::VectorXd
    //    solution_vector_l_linear = experiment_l_linear.ComputeSolution( kEps,
    //    kVelocity, kDirichletFunctor);
    //    solution_collection_wrapper_linear.final_time_solutions.push_back(solution_vector_l_linear);

    ecu_scheme::experiments::ConcentricStreamSolution<double>
        experiment_supg_l_linear(fe_space_l_linear);
    //    Eigen::VectorXd solution_vector_supg =
    //    ecu_scheme::assemble::SolveCDBVPSupg<decltype(kTempEps),
    //    decltype(kVelocity), decltype(kTestFunctor),
    //    decltype(kDirichletFunctor)>(fe_space_l_linear, kTempEps, kVelocity,
    //    kTestFunctor, kDirichletFunctor);
    //    solution_collection_wrapper_supg_linear.final_time_solutions.push_back(solution_vector_supg);
  }

  // Comparison with SUPG -- Linear FE case
  //  std::cout << "Comparison with SUPG - Linear FE space" << "\n";
  //  ecu_scheme::post_processing::convergence_comparison_toSUPG<double,
  //  decltype(mf_u_exact)>(
  //      solution_collection_wrapper_linear,
  //      solution_collection_wrapper_supg_linear, mf_u_exact,
  //      ecu_scheme::post_processing::concat(
  //          "concentric_stream_solution_comparison_linear", "_",
  //          refinement_levels, "_", eps_for_refinement),
  //      true
  //      );

  // COmparison with other methods - Quadratic FE space
  std::vector<
      std::pair<ecu_scheme::post_processing::ExperimentSolutionWrapper<double>,
                std::string>>
      bundle_solution_wrappers{
          //      {solution_collection_wrapper_quad, "UPWIND"},
          {solution_collection_wrapper_quad_stable, "7-Point Stable Upwind"},
          {solution_collection_wrapper_quad_fifteen_upwind, "15-Point Upwind"},
          {solution_collection_wrapper_supg_quad, "SUPG"}};
  ecu_scheme::post_processing::convergence_comparison_multiple_methods<
      double, decltype(mf_u_exact)>(
      bundle_solution_wrappers, mf_u_exact,
      ecu_scheme::post_processing::concat("concentric_stream_quad_comparison",
                                          "_", refinement_levels, "_",
                                          eps_for_refinement));
  return 0;
}