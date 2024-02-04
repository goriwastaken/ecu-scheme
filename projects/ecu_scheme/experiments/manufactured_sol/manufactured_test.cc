// Project header file include
#include "manufactured_solution.h"
// C system headers
// C++ standard library headers
#include <cmath>
// Other libraries headers
#include "lf/uscalfe/uscalfe.h"
#include "mesh.h"
#include "post_processing.h"

#include <Eigen/Core>

int main(int argc, char* argv[]){
  if(argc != 3 && argc != 1){
    std::cerr << "Usage: " << argv[0] << " refinement_levels eps " << std::endl;
    return -1;
  }
  // Diffusion coefficient for the regular refinement case
  double eps_for_refinement = 1e-8;
  // Number of refinement levels
  unsigned int refinement_levels = 3;
  // Adjust the number of refinement levels and the diffusion coefficient if the user specified them
  if(argc == 3){
    refinement_levels = std::stoi(argv[1]);
    eps_for_refinement = std::stod(argv[2]);
    if(eps_for_refinement < 1.0){
      std::cout << " good eps read " << eps_for_refinement << "\n";
    }
  }
  std::cout << "Refinement levels: " << refinement_levels << '\n';
  std::cout << "Epsilon for refinement: " << eps_for_refinement << '\n';

  ecu_scheme::mesh::BasicMeshBuilder builder;
  builder.SetNumCellsX(49);
  builder.SetNumCellsY(49);
  std::shared_ptr<lf::mesh::Mesh> mesh_p = builder.Build();

  auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);

//  const lf::assemble::DofHandler &dofh{fe_space->LocGlobMap()};

  // Manufactured solution diffusion coefficient - only used for plotting the solution
  const double kEps = eps_for_refinement;

  // Manufactured solution velocity field
  const auto kVelocity = [](const Eigen::Vector2d &x){
    return (Eigen::Vector2d() << 2.0, 3.0).finished();
  };


  // Manufactured exact solution u
  const auto kExactSolution = [eps_for_refinement](const Eigen::Vector2d &x){
    return x(0)*x(1)*x(1) - x(1)*x(1)*std::exp(2.0*(x(0)-1.0)/eps_for_refinement) - x(0)*std::exp(3.0*(x(1)-1.0)/eps_for_refinement) + std::exp(2.0*(x(0)-1.0)/eps_for_refinement + 3.0*(x(1)-1.0)/eps_for_refinement);
  };
  // Gradient of exact solution u
  const auto kGradExactSolution = [eps_for_refinement](const Eigen::Vector2d& x){
    Eigen::Vector2d grad;
    grad(0) = x(1)*x(1) - x(1)*x(1)*2.0/eps_for_refinement*std::exp(2.0*(x(0)-1.0)/eps_for_refinement) - std::exp(3.0*(x(1)-1.0)/eps_for_refinement) + 2.0/eps_for_refinement*std::exp(2.0*(x(0)-1.0)/eps_for_refinement + 3.0*(x(1)-1.0)/eps_for_refinement);
    grad(1) = 2*x(0)*x(1) - 2*x(1)*std::exp(2*(x(0)-1.0)/eps_for_refinement) -
              x(0) * 3.0/eps_for_refinement * std::exp(3*(x(1)-1.0)/eps_for_refinement) +
              3.0/eps_for_refinement * std::exp(2*(x(0)-1.0)/eps_for_refinement + 3*(x(1)-1.0)/eps_for_refinement);
    return grad;
  };
  // Laplacian of the exact solution
  const auto kLaplaceExactSolution = [eps_for_refinement](const Eigen::Vector2d& x){
    double const uxx = -4.0 / (eps_for_refinement*eps_for_refinement) * x(1)*x(1)*std::exp(2*(x(0)-1.0)/eps_for_refinement)+
                  4.0/(eps_for_refinement*eps_for_refinement)*std::exp(2*(x(0)-1.0)/eps_for_refinement + 3*(x(1)-1.0)/eps_for_refinement);
    double const uyy = 2.0*x(0) - 2.0*std::exp(2*(x(0)-1.0)/eps_for_refinement) -
                  x(0)*9.0/(eps_for_refinement*eps_for_refinement)*std::exp(3*(x(1)-1.0)/eps_for_refinement) +
                  9.0/(eps_for_refinement*eps_for_refinement)*std::exp(2*(x(0)-1.0)/eps_for_refinement + 3*(x(1)-1.0)/eps_for_refinement);
    return uxx + uyy;
  };
  // Compute the right hand side f correctly
  const auto kTestFunctor = [eps_for_refinement, kVelocity, kGradExactSolution, kLaplaceExactSolution](const Eigen::Vector2d &x){
    return -eps_for_refinement * kLaplaceExactSolution(x) + kVelocity(x).transpose() * kGradExactSolution(x);
  };
  // Compute the dirichlet function g correctly
  const auto kDirichletFunctor = [kExactSolution](const Eigen::Vector2d &x){
    return kExactSolution(x);
  };




  // Wrap all functors into LehrFEM MeshFunctions
  lf::mesh::utils::MeshFunctionConstant mf_eps{kEps};
  lf::mesh::utils::MeshFunctionGlobal mf_velocity{kVelocity};
  lf::mesh::utils::MeshFunctionGlobal mf_f_test_function{kTestFunctor};
  lf::mesh::utils::MeshFunctionGlobal mf_g_dirichlet{kDirichletFunctor};
  lf::mesh::utils::MeshFunctionGlobal<decltype(kExactSolution)> mf_exact_solution{kExactSolution};


  ecu_scheme::experiments::ManufacturedSolutionExperiment<double> experiment(fe_space);
  Eigen::VectorXd solution_vector = experiment.ComputeSolution(1e-8, kVelocity, kTestFunctor, kDirichletFunctor);
  // Plot of the computed solution
  ecu_scheme::post_processing::output_results<double>(fe_space, solution_vector, "manufactured_solution");

  // Compute the L2 error norms with regular refinement
  std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p = lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p, refinement_levels);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  multi_mesh.PrintInfo(std::cout);

  // Initialize solution wrapper with the determined refinement levels, the mesh hierarchy, and the yet empty vector of solutions
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double> solution_collection_wrapper{
      refinement_levels, eps_for_refinement, Eigen::VectorXd::Zero(refinement_levels), multi_mesh_p, std::vector<Eigen::VectorXd>{}
  };

  ecu_scheme::post_processing::ExperimentSolutionWrapper<double> solution_collection_wrapper_linear{
      refinement_levels, eps_for_refinement, Eigen::VectorXd::Zero(refinement_levels), multi_mesh_p, std::vector<Eigen::VectorXd>{}
  };

  ecu_scheme::post_processing::ExperimentSolutionWrapper<double> solution_collection_wrapper_supg{
        refinement_levels, eps_for_refinement, Eigen::VectorXd::Zero(refinement_levels), multi_mesh_p, std::vector<Eigen::VectorXd>{}
  };

  ecu_scheme::post_processing::ExperimentSolutionWrapper<double> solution_collection_wrapper_supg_quadratic{
        refinement_levels, eps_for_refinement, Eigen::VectorXd::Zero(refinement_levels), multi_mesh_p, std::vector<Eigen::VectorXd>{}
  };

  // get number of levels
  auto L = multi_mesh.NumLevels();
  LF_ASSERT_MSG(refinement_levels + 1 == L, "Number of levels in mesh hierarchy " << multi_mesh.NumLevels() << " is not equal to the number of refinement levels");
  for(int l = 0; l < L; ++l){
    std::cout << "Level " << l << " of manufactured sol" << '\n';
        // Compute finite element solution for every level and wrap every solution vector in a vector
        std::shared_ptr<const lf::mesh::Mesh> mesh_l{multi_mesh.getMesh(l)};
        auto fe_space_l = std::make_shared< lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_l);

        ecu_scheme::experiments::ManufacturedSolutionExperiment<double> experiment_l(fe_space_l);
        Eigen::VectorXd solution_vector_l = experiment_l.ComputeSolution(eps_for_refinement, kVelocity, kTestFunctor, kDirichletFunctor);
        solution_collection_wrapper.final_time_solutions.push_back(solution_vector_l);

        Eigen::VectorXd solution_vector_supg_quadratic = experiment_l.ComputeSUPGSolution(eps_for_refinement, kVelocity, kTestFunctor, kDirichletFunctor);
        solution_collection_wrapper_supg_quadratic.final_time_solutions.push_back(solution_vector_supg_quadratic);

        // only debug purposes - plot FE solution in paraview
        //ecu_scheme::post_processing::output_results<double>(fe_space_l, solution_vector_l, ecu_scheme::post_processing::concat("manufactured", "_", l, "_hierarchy"));
        //ecu_scheme::post_processing::output_meshfunction_paraview<double, decltype(mf_exact_solution)>(fe_space_l, mf_exact_solution, ecu_scheme::post_processing::concat("manuf_sol_",l,"_hierarchy"));

        // Compute finite element solution for linear Lagrangian finite elements
        auto fe_space_l_linear = std::make_shared< lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_l);
        ecu_scheme::experiments::ManufacturedSolutionExperiment<double> experiment_l_linear(fe_space_l_linear);
        Eigen::VectorXd solution_vector_l_linear = experiment_l_linear.ComputeSolution(eps_for_refinement, kVelocity, kTestFunctor, kDirichletFunctor);
        solution_collection_wrapper_linear.final_time_solutions.push_back(solution_vector_l_linear);

        // Compute SUPG linear FE solution
        const auto kTempEps = [kEps](const Eigen::Vector2d &x){return kEps;};
        Eigen::VectorXd solution_vector_supg = ecu_scheme::assemble::SolveCDBVPSupg<decltype(kTempEps), decltype(kVelocity), decltype(kTestFunctor), decltype(kDirichletFunctor)>(fe_space_l_linear, kTempEps, kVelocity, kTestFunctor, kDirichletFunctor);
        solution_collection_wrapper_supg.final_time_solutions.push_back(solution_vector_supg);


  }
  // Process the errors in the quadratic FE case
  std::cout << "Convergence of manufactured solution with quadratic FE\n";
  ecu_scheme::post_processing::convergence_report_single<
      double, decltype(mf_exact_solution)>(
      solution_collection_wrapper, mf_exact_solution,
      ecu_scheme::post_processing::concat("manufactured_solution_conv", "_",
                                          refinement_levels, "_",
                                          eps_for_refinement));
  // Process the errors in the linear FE case
  std::cout << "Convergence of manufactured solution with linear FE\n";
  ecu_scheme::post_processing::convergence_report_single<
      double, decltype(mf_exact_solution)>(
      solution_collection_wrapper_linear, mf_exact_solution,
      ecu_scheme::post_processing::concat("manufactured_solution_conv_linear",
                                          "_", refinement_levels, "_",
                                          eps_for_refinement),
      true);

  // Comparison with SUPG
  std::cout << "Comparison with SUPG -- Linear FE space " << "\n";
  ecu_scheme::post_processing::convergence_comparison_toSUPG<
      double, decltype(mf_exact_solution)>(
      solution_collection_wrapper_linear, solution_collection_wrapper_supg,
      mf_exact_solution,
      ecu_scheme::post_processing::concat(
          "manufactured_solution_conv_comparison", "_", refinement_levels, "_",
          eps_for_refinement),
      true);

  // Comparison with SUPG - Quadratic FE space
  std::cout << "Comparison with SUPG -- Quadratic FE space " << "\n";
  ecu_scheme::post_processing::convergence_comparison_toSUPG<
      double, decltype(mf_exact_solution)>(
      solution_collection_wrapper, solution_collection_wrapper_supg_quadratic,
      mf_exact_solution,
      ecu_scheme::post_processing::concat(
          "manufactured_solution_conv_comparison_quadratic", "_",
          refinement_levels, "_", eps_for_refinement));
  return 0;
}