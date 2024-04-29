// Project header file include
#include "const_velo_solution.h"
// C system headers
// C++ standard library headers
#include <iostream>
#include <fstream>
#include <cmath>
// Other libraries headers
#include "lf/uscalfe/uscalfe.h"
#include "mesh.h"
#include "post_processing.h"

/**
 * @brief Test of Convergence: Constant Velocity
 */
int main(int argc, char* argv[]){
  if(argc != 3 && argc != 1 && argc != 4){
    std::cerr << "Usage: " << argv[0] << " refinement_levels eps " << std::endl;
    return -1;
  }
  // Diffusion coefficient for the regular refinement case
  double eps_for_refinement = 0;
  // Number of refinement levels
  unsigned int refinement_levels = 7;
  // Time step size for linear FE space and quadratic FE space
  const double step_size_linear = 0.1;
  double step_size_quad = 0.01;
  // Adjust the number of refinement levels and the diffusion coefficient if the user specified them
  if(argc == 3){
    refinement_levels = std::stoi(argv[1]);
    eps_for_refinement = std::stod(argv[2]);
  }else if(argc == 4){
    refinement_levels = std::stoi(argv[1]);
    eps_for_refinement = std::stod(argv[2]);
    step_size_quad = std::stod(argv[3]);
  }
  std::cout << "Refinement levels: " << refinement_levels << '\n';
  std::cout << "Epsilon for refinement: " << eps_for_refinement << '\n';
  std::cout << "Step size for quadratic FE space: " << step_size_quad << '\n';

  // setup is [0,2]x[0,2] with periodic boundary conditions and time interval I= [0,0.5]
  // initial condition A_0 = 1/pi * cos(pi*y) + 1/(2pi) * cos(2pi*x)
  // velocity field u = (4,4)
  // forcing term f = 0
  // use Heun time-stepping with uniform time step delta_t = 0.1*h for bilinear Lagrangian FE and delta_t = 0.01*h for biquadratic Lagrangian FE
  // compare solution at final time t=0.5 with initial condition
  // report L2 and H1 error norms

  ecu_scheme::mesh::BasicMeshBuilder builder;
  builder.SetNumCellsX(2);
  builder.SetNumCellsY(2);
  std::shared_ptr<lf::mesh::Mesh> mesh_p = builder.Build(0.0, 0.0, 1.0, 1.0);

  auto fe_space_quad = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
  auto fe_space_linear = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);


  // initial condition for the experiment
//  const auto kAInitCondition = [](const Eigen::Vector2d &x){
//    return 1.0/M_PI * std::cos(M_PI*x(1)) + 1.0/(2.0*M_PI) * std::cos(2.0*M_PI*x(0));
//  };
//  // constant velocity field
//  const auto kVelocity = [](const Eigen::Vector2d &x){
//    return (Eigen::Vector2d() << 4.0, 4.0).finished();
//  };
//  // forcing term f
//  const auto kTestFunctor = [](const Eigen::Vector2d &x){
//    return 0.0;
//  };
//  const auto kPeriodicBC = [kAInitCondition](const Eigen::Vector2d& x){
//    if(x(0) < 1e-5 || x(0) > (2.0 - 1e-5) || x(1) < 1e-5 || x(1) > (2.0 - 1e-5)){
//      return kAInitCondition(x);
//    }
//    return 0.0;
//  };

  // new setup stationary constant velocity problem
  const auto kVelocity = [](const Eigen::Vector2d& x){
    return (Eigen::Vector2d() << 1.0, 1.0).finished();
  };
  const auto kBumpFunction = [](const double &xi){
    return 0.5*std::cos(0.5*M_PI*xi)*std::cos(0.5*M_PI*xi);
  };
  const auto kDirichletFunction = [kBumpFunction](const Eigen::Vector2d& x){
    const double x_norm = x.norm();
    if(x_norm >= 1.0){
      return 0.0;
    }
    return kBumpFunction(x_norm);
  };

  const auto kTestFunctor = [](const Eigen::Vector2d& x){
    const double diff = x(0) - x(1);
    const double u_x = std::cos(0.5*M_PI*diff)*std::sin(0.5*M_PI*diff)*(0.5*M_PI);
    const double u_y = std::cos(0.5*M_PI*diff)*std::sin(0.5*M_PI*diff)*(-0.5*M_PI);
    return u_x + u_y;
  };
//  const auto kExactSolution = [kDirichletFunction](const Eigen::Vector2d& x){
//    return kDirichletFunction(x);
//  };
  const auto kExactSolution = [kBumpFunction](const Eigen::Vector2d& x){
    return kBumpFunction(x(0) - x(1));
  };



  // Wrap all functors into LehrFEM MeshFunctions
//  lf::mesh::utils::MeshFunctionGlobal mf_a_init{kAInitCondition};
  lf::mesh::utils::MeshFunctionGlobal mf_a_init{kDirichletFunction};
  lf::mesh::utils::MeshFunctionGlobal mf_velocity{kVelocity};
  // we just consider the pure advection case
  lf::mesh::utils::MeshFunctionGlobal mf_f_test_function{kTestFunctor};
  lf::mesh::utils::MeshFunctionGlobal mf_exact_solution{kExactSolution};


  //Eigen::VectorXd solution_vector = Eigen::VectorXd::Zero(1);

  // Experiment for the quadratic FE space case
  const double max_time_quad = 0.5;
//  ecu_scheme::experiments::ConstVeloSolution<double> experiment(fe_space_quad);
//  std::vector<Eigen::VectorXd> solution_vector = experiment.ComputeSolution(kAInitCondition, kVelocity, kPeriodicBC, max_time_quad, step_size_quad);

  // Get the solution after the first timestep
//  Eigen::VectorXd solution_vector_first_timestep = solution_vector[0];
  // Get the solution at the final time
//  Eigen::VectorXd solution_vector_final_timestep = solution_vector[solution_vector.size() - 1];

//  lf::mesh::utils::MeshFunctionGlobal mf_sol_first_timestep{solution_vector_first_timestep};
//  lf::mesh::utils::MeshFunctionGlobal mf_sol_final_timestep{solution_vector_final_timestep};

  //ecu_scheme::post_processing::output_results<double>(fe_space_quad, solution_vector_final_timestep, "const_velo_solution_quad");
  // Projection of exact and numerical solution on the curve {x \in \Omega, y = 1}
//  auto horizontalCurve = [](double t) -> Eigen::Vector2d{
//    return Eigen::Vector2d(t, 1.0);
//  };
//  ecu_scheme::post_processing::SampleMeshFunctionOnCurve(mesh_p, horizontalCurve, mf_a_init, 300,  "const_velo_linear_projection_curve", {0.0, 2.0});

  std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p = lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p, refinement_levels);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  multi_mesh.PrintInfo(std::cout);

  // Initialize solution wrapper with the determined refinement levels, the mesh hierarchy, and a yet empty vector of solutions
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double> solution_collection_wrapper{
      refinement_levels, eps_for_refinement, Eigen::VectorXd::Zero(refinement_levels) , multi_mesh_p, std::vector<Eigen::VectorXd>{}
  };
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double> solution_collection_wrapper_stable{
      refinement_levels, eps_for_refinement, Eigen::VectorXd::Zero(refinement_levels) , multi_mesh_p, std::vector<Eigen::VectorXd>{}
  };
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double> solution_collection_wrapper_supg_quad{
      refinement_levels, eps_for_refinement, Eigen::VectorXd::Zero(refinement_levels) , multi_mesh_p, std::vector<Eigen::VectorXd>{}
  };

  // get number of levels
  auto L = multi_mesh.NumLevels();
  std::cout <<"checking multimesh levels " << multi_mesh.NumLevels() << std::endl;
  LF_ASSERT_MSG(L == refinement_levels + 1, "Number of levels in mesh hierarchy is not equal to the number of refinement levels");
  for(int l = 0; l < L; ++l){
    // Compute finite element solution for every level and wrap every solution vector in a vector
    auto mesh_l = multi_mesh.getMesh(l);
    auto fe_space_l = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_l);
    // Compute maximal mesh width for each level and add it to the Wrapper
//    for(const lf::mesh::Entity *e : mesh_l->Entities(1)){
//      const double h_temp = lf::geometry::Volume(*e->Geometry());
//      if(h_temp > solution_collection_wrapper.max_meshwidth_per_level(l)){
//        solution_collection_wrapper.max_meshwidth_per_level(l) = h_temp;
//      }
//    }
          // Compute the solution for the current level
    ecu_scheme::experiments::ConstVeloSolution<double> experiment(fe_space_l);
//    Eigen::VectorXd solution_vector_quad = experiment.ComputeSolution(kDirichletFunction, kVelocity, kTestFunctor, "UPWIND");
//    solution_collection_wrapper.final_time_solutions.push_back(solution_vector_quad);

    // Compute stable upwind solution with 7-point quad rule
    Eigen::VectorXd solution_vector_stable_quad = experiment.ComputeSolution(kDirichletFunction, kVelocity, kTestFunctor, "STABLE_UPWIND");
    solution_collection_wrapper_stable.final_time_solutions.push_back(solution_vector_stable_quad);

    // Compute SUPG solution
    const auto kTempEps = [eps_for_refinement](const Eigen::Vector2d& x){return eps_for_refinement;};
    Eigen::VectorXd solution_vector_supg_quad = ecu_scheme::assemble::SolveCDBVPSupgQuad<decltype(kTempEps), decltype(kVelocity), decltype(kTestFunctor), decltype(kDirichletFunction)>(fe_space_l, kTempEps, kVelocity, kTestFunctor, kDirichletFunction);
    solution_collection_wrapper_supg_quad.final_time_solutions.push_back(solution_vector_supg_quad);
  }

  // Realize convergence results for uniform refinement
//  ecu_scheme::post_processing::convergence_report_single<double>(solution_collection_wrapper, mf_exact_solution,
//                                                                 ecu_scheme::post_processing::concat("const_velo_quad","_",refinement_levels,"_",eps_for_refinement));
  std::cout << "Comparison with other methods - Quadratic FE space" << "\n";
  std::vector<std::pair<ecu_scheme::post_processing::ExperimentSolutionWrapper<double>, std::string>> bundle_solution_wrappers{
//            {solution_collection_wrapper, "UPWIND"},
      {solution_collection_wrapper_stable, "7-Point Stable Upwind"},
//      {solution_collection_wrapper_quad_fifteen_upwind, "15P_UPWIND"},
      {solution_collection_wrapper_supg_quad, "SUPG"}
  };
  ecu_scheme::post_processing::convergence_comparison_multiple_methods<double, decltype(mf_exact_solution)>(
      bundle_solution_wrappers,
      mf_exact_solution,
      ecu_scheme::post_processing::concat(
          "const_velo_multiple_methods_quad", "_", refinement_levels, "_",
          eps_for_refinement)
  );
  return 0;
}