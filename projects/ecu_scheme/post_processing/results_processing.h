
#ifndef THESIS_POST_PROCESSING_RESULTS_PROCESSING_H_
#define THESIS_POST_PROCESSING_RESULTS_PROCESSING_H_

#include <lf/fe/fe.h>
#include <lf/uscalfe/uscalfe.h>
#include <lf/io/io.h>
#include <lf/refinement/refinement.h>
#include "norms.h"

#include <string>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <utility>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
namespace ecu_scheme::post_processing {

//todo wrappers for solution vectors
//todo method for comparing results

/**
 * @brief Concatenate objects defining an operator<<(std::ostream&)
 *
 * C++17 specific implementation based on fold expressions
 * Reference: https://stackoverflow.com/a/61002819
 *
 * @tparam Args variadic template argument pack
 * @param args variadic pack of objects implementing operator<<(std::ostream&)
 * @return string containing the concatenation of the objects in args
 */
template<typename... Args>
static std::string concat(Args &&...args){
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return oss.str();
}

template<typename SCALAR, typename MF>
void output_meshfunction_paraview(std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space,
                                  MF& mf_analytic,
                                  std::string file_name){
  //generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir);

  // output results to vtk file
  lf::io::VtkWriter vtk_writer(fe_space->Mesh(), concat(main_dir, "/", file_name, ".vtk"));
  vtk_writer.WritePointData(concat(file_name, "_solution"), mf_analytic);
}


template<typename SCALAR>
void output_results(const std::shared_ptr< lf::fe::ScalarFESpace<SCALAR>>& fe_space,
                    Eigen::VectorXd solution_vec,
                    std::string experiment_name){

  //generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir);

  //output results to vtk file
  lf::fe::MeshFunctionFE mf_sol(fe_space, solution_vec);
  lf::io::VtkWriter vtk_writer(fe_space->Mesh(), concat(main_dir, "/", experiment_name, ".vtk"), 0, 2);
  vtk_writer.WritePointData(concat(experiment_name, "_solution_high"), mf_sol);
}

template<typename SCALAR, typename MF>
void convergence_report_single(const ExperimentSolutionWrapper<SCALAR>& solution_collection_wrapper,
                         MF& mf_exact_solution,
                         std::string experiment_name, bool isLinear = false){
  // generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir); //if necessary

//  // set Eigen csv format if needed
//  const Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
//                                  Eigen::DontAlignCols, ", ", "\n");

  // create a csv file to store the results for plotting
  std::ofstream L2norm_csv_file;
  L2norm_csv_file.open(concat(main_dir, "/", experiment_name, "_L2error.csv"));
  // include metadata about the experiment such as number of refinement levels and the diffusion coefficient used
  L2norm_csv_file << "Number of refinement levels: " << solution_collection_wrapper.refinement_levels << "\n";
  L2norm_csv_file << "Diffusion coefficient: " << solution_collection_wrapper.eps << "\n";
  L2norm_csv_file << "No. of dofs,Meshwidth,L2 error" << "\n";

  // define square functions for norms
  auto square_scalar = [](SCALAR x) -> double{
    return std::abs(x) * std::abs(x);
  };
  auto square_vector = [](Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> x) -> double{
    return x.squaredNorm();
  };

  // define quadrature rule for norms
  lf::quad::QuadRule qr = lf::quad::make_QuadRule(lf::base::RefEl::kTria(), 4);

  // solution_collection_wrapper contains the mesh hierarchy used and hence each mesh corresponding to a refinement level
  lf::refinement::MeshHierarchy& multi_mesh{*solution_collection_wrapper.mesh_hierarchy_p};
  std::cout << "Results processing mesh hierarchy info: \n";
  multi_mesh.PrintInfo(std::cout);
  // get number of levels
  auto L = multi_mesh.NumLevels();

  Eigen::VectorXd Ndof_array(L);
  Eigen::VectorXd errors_array(L);
  // perform computations on all levels
  for(int l=0; l < L; ++l){
    // get mesh for refinement level l
    std::shared_ptr<const lf::mesh::Mesh> mesh_l{multi_mesh.getMesh(l)};
    // compute meshwidth
    double kHMax = 0.0;
    for(const lf::mesh::Entity* e : mesh_l->Entities(1)){
      double kH = lf::geometry::Volume(*(e->Geometry()));
      if(kH > kHMax){
        kHMax = kH;
      }
    }
    // get finite element space for refinement level l
    std::shared_ptr< lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space;
    if(isLinear){
      // Linear FE space
      fe_space = std::make_shared< lf::uscalfe::FeSpaceLagrangeO1<SCALAR>>(mesh_l);
    }else{
      // Quadratic FE space
      fe_space = std::make_shared< lf::uscalfe::FeSpaceLagrangeO2<SCALAR>>(mesh_l);
    }
    // get solution vector computed at refinement l
    Eigen::VectorXd solution_vec = solution_collection_wrapper.final_time_solutions.at(l);

    // Take finite element solution and wrap it into a mesh function
    auto mf_fe_sol = lf::fe::MeshFunctionFE(fe_space, solution_vec);

    // Compute the L2 error
    double L2_error = std::sqrt(lf::fe::IntegrateMeshFunction(*(fe_space->Mesh()), lf::mesh::utils::squaredNorm(mf_fe_sol - mf_exact_solution), 10));
    //double L2_error = std::sqrt(lf::fe::IntegrateMeshFunction(*(fe_space->Mesh()), lf::mesh::utils::squaredNorm(mf_fe_sol - mf_exact_solution), [](const lf::mesh::Entity& e){return lf::quad::make_QuadRule(e.RefEl(),4);}));
    Ndof_array(l) = fe_space->LocGlobMap().NumDofs();
    errors_array(l) = L2_error;


    //debug only - plot each mesh with matlab
    //lf::io::writeMatlab(*(fe_space->Mesh()), concat(main_dir, "/", experiment_name, "_matlabmesh", std::to_string(l), ".m"));
    // how to plot using lehrfem plot_

    // Add results to L2norm_csv_file file
    L2norm_csv_file << fe_space->LocGlobMap().NumDofs() << "," << kHMax << "," << L2_error << "\n";
    std::cout << std::left << std::setw(10) << fe_space->LocGlobMap().NumDofs() << std::left <<std::setw(16) << L2_error  << std::endl; //debug purpose
  }
  L2norm_csv_file.close();

  // Plot the computed L2error
  double eoc_value = eoc(Ndof_array, errors_array);
  std::cout << "EOC value: " << eoc_value << "\n";

}

template<typename SCALAR, typename MF>
void convergence_comparison_toSUPG(const ExperimentSolutionWrapper<SCALAR>& solution_collection_wrapper_one,
                            const ExperimentSolutionWrapper<SCALAR>& solution_collection_wrapper_two,
                            MF& mf_exact_solution, std::string experiment_name,
                            bool isLinear= false){
  // generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir); //if necessary

  // create a csv file to store the results for plotting
  std::ofstream L2norm_csv_file;
  L2norm_csv_file.open(concat(main_dir, "/", experiment_name, "_L2error.csv"));
  // include metadata about the experiment such as number of refinement levels and the diffusion coefficient used
  L2norm_csv_file << "Number of refinement levels: " << solution_collection_wrapper_one.refinement_levels << "\n";
  L2norm_csv_file << "Diffusion coefficient: " << solution_collection_wrapper_one.eps << "\n";
  L2norm_csv_file << "No. of dofs,Meshwidth,L2 error Upwind,L2 error SUPG" << "\n";

  // solution_collection_wrapper contains the mesh hierarchy used and hence each mesh corresponding to a refinement level
  lf::refinement::MeshHierarchy& multi_mesh{*solution_collection_wrapper_one.mesh_hierarchy_p};
  std::cout << "Results processing mesh hierarchy info: \n";
  multi_mesh.PrintInfo(std::cout);
  // get number of levels
  auto L = multi_mesh.NumLevels();

  for(int l=0; l < L; ++l){
    // get mesh for refinement level l
    std::shared_ptr<const lf::mesh::Mesh> mesh_l{multi_mesh.getMesh(l)};
    // compute meshwidth
    double kHMax = 0.0;
    for(const lf::mesh::Entity* e : mesh_l->Entities(1)){
      double kH = lf::geometry::Volume(*(e->Geometry()));
      if(kH > kHMax){
        kHMax = kH;
      }
    }
    // get finite element space for refinement level l
    std::shared_ptr< lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space;
    if(isLinear){
      // Linear FE space
      fe_space = std::make_shared< lf::uscalfe::FeSpaceLagrangeO1<SCALAR>>(mesh_l);
    }else{
      // Quadratic FE space
      fe_space = std::make_shared< lf::uscalfe::FeSpaceLagrangeO2<SCALAR>>(mesh_l);
    }
    // get solution vector computed at refinement l
    Eigen::VectorXd solution_vec_one = solution_collection_wrapper_one.final_time_solutions.at(l);
    Eigen::VectorXd solution_vec_two = solution_collection_wrapper_two.final_time_solutions.at(l);

    // Take finite element solution and wrap it into a mesh function
    auto mf_fe_sol_one = lf::fe::MeshFunctionFE(fe_space, solution_vec_one);
    auto mf_fe_sol_two = lf::fe::MeshFunctionFE(fe_space, solution_vec_two);

    // Compute the L2 error
    double L2_error_one = std::sqrt(lf::fe::IntegrateMeshFunction(*(fe_space->Mesh()), lf::mesh::utils::squaredNorm(mf_fe_sol_one - mf_exact_solution), 10));
    double L2_error_two = std::sqrt(lf::fe::IntegrateMeshFunction(*(fe_space->Mesh()), lf::mesh::utils::squaredNorm(mf_fe_sol_two - mf_exact_solution), 10));
    // Add results to L2norm_csv_file file
    L2norm_csv_file << fe_space->LocGlobMap().NumDofs() << "," << kHMax << "," << L2_error_one << "," << L2_error_two << "\n";
    std::cout << std::left << std::setw(10) << fe_space->LocGlobMap().NumDofs() << std::left <<std::setw(16) << L2_error_one << std::left <<std::setw(16) << L2_error_two << std::endl; //debug purpose
  }
}

} // post_processing

#endif //THESIS_POST_PROCESSING_RESULTS_PROCESSING_H_
