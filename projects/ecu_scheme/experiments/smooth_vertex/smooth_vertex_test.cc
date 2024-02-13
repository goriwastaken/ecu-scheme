#include "smooth_vertex_solution.h"

#include "lf/uscalfe/uscalfe.h"
#include "mesh.h"
#include "post_processing.h"


#include <iostream>

#include <memory>

/**
 * @brief Test of Accuracy for MHD: Smooth Vertex experiment from the dissertation Cecilia Pagliantini. Computational Magnetohydrodynamics with Discrete Differential Forms. PhD thesis, ETH Zürich, 2016. Chapter 4.3.1
 */
int main(int argc, char* argv[]){
  if(argc != 3 && argc != 1 && argc != 4){
    std::cerr << "Usage: " << argv[0] << " refinement_levels eps " << std::endl;
    return -1;
  }
  // Diffusion coefficient for the regular refinement case
  double eps_for_refinement = 0;
  // Number of refinement levels
  unsigned int refinement_levels = 2;
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

  // setup is [-5,5]x[-5,5] with periodic boundary conditions and time interval I= [0,0.5]

  ecu_scheme::mesh::BasicMeshBuilder builder;
  builder.SetNumCellsX(49);
  builder.SetNumCellsY(49);
  std::shared_ptr<lf::mesh::Mesh> mesh_p = builder.Build(-5.0, -5.0, 5.0, 5.0);

  // Define the functions - which are known analytically
  const auto kInitVelocity = [](const Eigen::Vector2d& x)-> Eigen::Vector2d{
    return Eigen::Vector2d(1.0, 1.0);
  };

  const auto radius = [](const Eigen::Vector3d& xh){
    return sqrt((xh(0) - 1.0*xh(2))*(xh(0) - 1.0*xh(2)) + (xh(1) - 1.0*xh(2))*(xh(1) - 1.0*xh(2)));
  };

  const auto kVelocity = [radius, kInitVelocity](const Eigen::Vector3d& xh){
    // xh has components (x, y, t)
    Eigen::Vector2d temp = (Eigen::Vector2d() << xh(2) - xh(1), xh(0) - xh(2)).finished();
    const double coefficient = 1.0/(M_PI*2.0) * exp(1.0/(2.0*(1.0 - radius(xh)*radius(xh))));
    const Eigen::Vector2d result =
        kInitVelocity((Eigen::Vector2d() << xh(0),xh(1)).finished()) +
                                  coefficient * temp;
    return result;
  };
  const auto kMagneticInductionExactSolution = [radius](const Eigen::Vector3d& xh){
    // xh has components (x, y, t)
    Eigen::Vector2d temp = (Eigen::Vector2d() << xh(2) - xh(1), xh(0) - xh(2)).finished();
    const double coefficient = 1.0/(M_PI*2.0) * exp(1.0/(2.0*(1.0 - radius(xh)*radius(xh))));
    const Eigen::Vector2d result = coefficient * temp;
    return result;
  };
  const auto kPotentialExactSolution = [radius](const Eigen::Vector3d& xh){
    // xh has components (x, y, t)
    return 1.0/(M_PI*2.0) * exp(1.0/(2.0*(1.0 - radius(xh)*radius(xh))));
  };

  const double kMaxTime = 0.5;

  auto fe_space_quad = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
  auto fe_space_linear = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);

  // Accuracy test by regular refinement
  std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p, refinement_levels);
  lf::refinement::MeshHierarchy& multi_mesh{*mesh_hierarchy};
  multi_mesh.PrintInfo(std::cout);

  // Initialize solution wrappers for linear and quadratic FE
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double> solution_collection_wrapper_linear{
      refinement_levels, eps_for_refinement, Eigen::VectorXd::Zero(refinement_levels), mesh_hierarchy, std::vector<Eigen::VectorXd>{}
  };
  ecu_scheme::post_processing::ExperimentSolutionWrapper<double> solution_collection_wrapper_quad{
      refinement_levels, eps_for_refinement, Eigen::VectorXd::Zero(refinement_levels), mesh_hierarchy, std::vector<Eigen::VectorXd>{}
  };

  // get number of levels and iterate through them to check convergence
  auto L = multi_mesh.NumLevels();
  for(int l = 0; l < L; ++l){
    // Compute FE solution for every level for linear and quadratic FE space
    auto mesh_l = multi_mesh.getMesh(l);
    // Quadratic case
    auto fe_space_l_quad = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_l);
    //todo find max meshwidth
    //Compute the solution for the current level
    ecu_scheme::experiments::SmoothVertexSolution<double> experiment_quad(fe_space_l_quad);
    std::vector<Eigen::VectorXd> solution_vector_quad = experiment_quad.ComputeSolution(kVelocity, kMagneticInductionExactSolution, kPotentialExactSolution, kMaxTime, step_size_quad);
    solution_collection_wrapper_quad.final_time_solutions.push_back(solution_vector_quad.at(solution_vector_quad.size() - 1));
    // Linear case
    auto fe_space_l_lin = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_l);
  }

  // Realize convergence report for uniform refinement
//  ecu_scheme::post_processing::convergence_report_single<double>(
//      solution_collection_wrapper_quad, kPotentialExactSolution, "smooth_vertex_solution_conv_quad");

}
