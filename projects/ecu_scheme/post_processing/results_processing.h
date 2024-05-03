
#ifndef THESIS_POST_PROCESSING_RESULTS_PROCESSING_H_
#define THESIS_POST_PROCESSING_RESULTS_PROCESSING_H_

#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>

#include "ecu_tools.h"
#include "norms.h"
namespace ecu_scheme::post_processing {

/**
 * @brief Wrapper for storing the solutions at different refinement levels of an
 * experiment
 * @tparam SCALAR base type of the solution vector
 */
template <typename SCALAR>
struct ExperimentSolutionWrapper {
  unsigned int refinement_levels;
  double eps;
  Eigen::VectorXd max_meshwidth_per_level;
  std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy_p;
  std::vector<Eigen::VectorXd> final_time_solutions;
};

Eigen::Vector2d linearFit(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
  assert(x.rows() == y.rows());
  Eigen::Matrix<double, Eigen::Dynamic, 2> X(x.rows(), 2);

  X.col(0) = Eigen::VectorXd::Constant(x.rows(), 1.0);
  X.col(1) = x;

  return X.fullPivHouseholderQr().solve(y);
}

double eoc(Eigen::VectorXd& N, Eigen::VectorXd& err,
           unsigned int from_index = 0) {
  const unsigned dim = N.size();

  // truncate pre-asymptotic behaviour if desired
  const unsigned int newdim = dim - from_index;
  // compute log(N) and log(err) component-wise
  auto logfun = [](double d) { return std::log(d); };
  Eigen::VectorXd Nlog(newdim), errlog(newdim);
  std::transform(N.data() + from_index, N.data() + dim, Nlog.data(), logfun);
  std::transform(err.data() + from_index, err.data() + dim, errlog.data(),
                 logfun);

  Eigen::Vector2d polyfit = linearFit(Nlog, errlog);
  double alpha = -polyfit[1];

  return alpha;
}

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
template <typename... Args>
static std::string concat(Args&&... args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return oss.str();
}

template <typename SCALAR, typename MF>
void output_meshfunction_paraview(
    std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space,
    MF& mf_analytic, std::string file_name) {
  // generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir);

  // output results to vtk file
  lf::io::VtkWriter vtk_writer(fe_space->Mesh(),
                               concat(main_dir, "/", file_name, ".vtk"));
  vtk_writer.WritePointData(concat(file_name, "_solution"), mf_analytic);
}

template <typename SCALAR>
void output_results(
    const std::shared_ptr<lf::fe::ScalarFESpace<SCALAR>>& fe_space,
    Eigen::VectorXd solution_vec, std::string experiment_name) {
  // generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir);

  // output results to vtk file
  lf::fe::MeshFunctionFE mf_sol(fe_space, solution_vec);
  lf::io::VtkWriter vtk_writer(
      fe_space->Mesh(), concat(main_dir, "/", experiment_name, ".vtk"), 0, 2);
  vtk_writer.WritePointData(concat(experiment_name, "_solution_high"), mf_sol);
}

template <typename SCALAR, typename MF>
void convergence_report_oneform(
    const ExperimentSolutionWrapper<SCALAR>& solution_collection_wrapper,
    MF& mf_exact_solution, std::string experiment_name, bool isLinear = false) {
  // generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir);  // if necessary

  //  // set Eigen csv format if needed
  //  const Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
  //                                  Eigen::DontAlignCols, ", ", "\n");

  // create a csv file to store the results for plotting
  std::ofstream L2norm_csv_file;
  L2norm_csv_file.open(concat(main_dir, "/", experiment_name, "_L2error.csv"));
  // include metadata about the experiment such as number of refinement levels
  // and the diffusion coefficient used
  L2norm_csv_file << "Number of refinement levels: "
                  << solution_collection_wrapper.refinement_levels << "\n";
  L2norm_csv_file << "Diffusion coefficient: "
                  << solution_collection_wrapper.eps << "\n";
  L2norm_csv_file << "No. of dofs,Meshwidth,Extrusion-Contraction Upwind"
                  << "\n";

  // define square functions for norms
  auto square_scalar = [](SCALAR x) -> double {
    return std::abs(x) * std::abs(x);
  };
  auto square_vector =
      [](Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> x) -> double {
    return x.squaredNorm();
  };

  // solution_collection_wrapper contains the mesh hierarchy used and hence each
  // mesh corresponding to a refinement level
  lf::refinement::MeshHierarchy& multi_mesh{
      *solution_collection_wrapper.mesh_hierarchy_p};

  // get number of levels
  auto L = multi_mesh.NumLevels();

  Eigen::VectorXd Ndof_array(L);
  Eigen::VectorXd errors_array(L);
  // perform computations on all levels
  for (int l = 0; l < L; ++l) {
    // get mesh for refinement level l
    std::shared_ptr<const lf::mesh::Mesh> mesh_l{multi_mesh.getMesh(l)};

    // generate DOFHandler corresponding to edge element basis functions to
    // report number of DOFs
    const lf::assemble::DofHandler& dofh_edge =
        isLinear ? lf::assemble::UniformFEDofHandler(
                       mesh_l, {{lf::base::RefEl::kSegment(), 1}})
                 : lf::assemble::UniformFEDofHandler(
                       mesh_l, {{lf::base::RefEl::kSegment(), 2},
                                {lf::base::RefEl::kTria(), 2}});
    // compute meshwidth
    const double kHMax =
        ecu_scheme::post_processing::ComputeMeshWidthTria(mesh_l);
    // get finite element space for refinement level l
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space;
    if (isLinear) {
      // Linear FE space
      fe_space =
          std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<SCALAR>>(mesh_l);
    } else {
      // Quadratic FE space
      fe_space =
          std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<SCALAR>>(mesh_l);
    }
    // get solution vector computed at refinement l
    Eigen::VectorXd solution_vec =
        solution_collection_wrapper.final_time_solutions.at(l);

    // Take finite element solution and wrap it into specialized mesh function
    // for 1-forms
    ecu_scheme::assemble::MeshFunctionOneForm<double> mf_fe_one_form(
        solution_vec, mesh_l);
    // Compute difference with exact solution (care that the
    // lf::uscalfe::operator-() is used
    auto mf_diff = mf_fe_one_form - mf_exact_solution;

    // Prepare computation of L2-error, generate a quadrature rule
    lf::quad::QuadRule quad_rule =
        lf::quad::make_QuadRule(lf::base::RefEl::kTria(), 10);
    // Compute L2 error
    const std::pair<double, lf::mesh::utils::CodimMeshDataSet<double>>
        L2_error_bundle = ecu_scheme::post_processing::L2norm(
            mesh_l, mf_diff, square_vector, quad_rule);

    Ndof_array(l) = dofh_edge.NumDofs();
    errors_array(l) = std::get<0>(L2_error_bundle);

    // debug only - plot each mesh with matlab
    // lf::io::writeMatlab(*(fe_space->Mesh()), concat(main_dir, "/",
    // experiment_name, "_matlabmesh", std::to_string(l), ".m"));
    //  how to plot using lehrfem plot_

    // Add results to L2norm_csv_file file
    L2norm_csv_file << dofh_edge.NumDofs() << "," << kHMax << ","
                    << std::get<0>(L2_error_bundle) << "\n";
    std::cout << std::left << std::setw(10) << fe_space->LocGlobMap().NumDofs()
              << std::left << std::setw(16) << dofh_edge.NumDofs() << std::left
              << std::setw(16) << std::get<0>(L2_error_bundle)
              << std::endl;  // debug purpose
  }
  L2norm_csv_file.close();

  // Plot the computed L2error rate
  double eoc_value = eoc(Ndof_array, errors_array);
  std::cout << "EOC value: " << eoc_value << "\n";
}

template <typename SCALAR, typename MF>
void convergence_comparison_toSUPG(
    const ExperimentSolutionWrapper<SCALAR>& solution_collection_wrapper_one,
    const ExperimentSolutionWrapper<SCALAR>& solution_collection_wrapper_two,
    MF& mf_exact_solution, std::string experiment_name, bool isLinear = false) {
  // generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir);  // if necessary

  // create a csv file to store the results for plotting
  std::ofstream L2norm_csv_file;
  L2norm_csv_file.open(concat(main_dir, "/", experiment_name, "_L2error.csv"));
  // include metadata about the experiment such as number of refinement levels
  // and the diffusion coefficient used
  L2norm_csv_file << "Number of refinement levels: "
                  << solution_collection_wrapper_one.refinement_levels << "\n";
  L2norm_csv_file << "Diffusion coefficient: "
                  << solution_collection_wrapper_one.eps << "\n";
  L2norm_csv_file << "No. of dofs,Meshwidth,Midpoint Upwind,SUPG"
                  << "\n";

  // solution_collection_wrapper contains the mesh hierarchy used and hence each
  // mesh corresponding to a refinement level
  lf::refinement::MeshHierarchy& multi_mesh{
      *solution_collection_wrapper_one.mesh_hierarchy_p};
  std::cout << "Results processing mesh hierarchy info: \n";
  multi_mesh.PrintInfo(std::cout);
  // get number of levels
  auto L = multi_mesh.NumLevels();

  Eigen::VectorXd Ndof_array(L);
  Eigen::VectorXd errors_array(L);

  for (int l = 0; l < L; ++l) {
    // get mesh for refinement level l
    std::shared_ptr<const lf::mesh::Mesh> mesh_l{multi_mesh.getMesh(l)};
    // compute meshwidth
    const double kHMax =
        ecu_scheme::post_processing::ComputeMeshWidthTria(mesh_l);
    // get finite element space for refinement level l
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space;
    if (isLinear) {
      // Linear FE space
      fe_space =
          std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<SCALAR>>(mesh_l);
    } else {
      // Quadratic FE space
      fe_space =
          std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<SCALAR>>(mesh_l);
    }
    // get solution vector computed at refinement l
    Eigen::VectorXd solution_vec_one =
        solution_collection_wrapper_one.final_time_solutions.at(l);
    Eigen::VectorXd solution_vec_two =
        solution_collection_wrapper_two.final_time_solutions.at(l);

    // Take finite element solution and wrap it into a mesh function
    auto mf_fe_sol_one = lf::fe::MeshFunctionFE(fe_space, solution_vec_one);
    auto mf_fe_sol_two = lf::fe::MeshFunctionFE(fe_space, solution_vec_two);

    // Compute the L2 error
    double L2_error_one = std::sqrt(lf::fe::IntegrateMeshFunction(
        *(fe_space->Mesh()),
        lf::mesh::utils::squaredNorm(mf_fe_sol_one - mf_exact_solution), 4));
    double L2_error_two = std::sqrt(lf::fe::IntegrateMeshFunction(
        *(fe_space->Mesh()),
        lf::mesh::utils::squaredNorm(mf_fe_sol_two - mf_exact_solution), 4));

    Ndof_array(l) = fe_space->LocGlobMap().NumDofs();
    errors_array(l) = L2_error_one;
    // Add results to L2norm_csv_file file
    L2norm_csv_file << fe_space->LocGlobMap().NumDofs() << "," << kHMax << ","
                    << L2_error_one << "," << L2_error_two << "\n";
    std::cout << std::left << std::setw(10) << fe_space->LocGlobMap().NumDofs()
              << std::left << std::setw(16) << L2_error_one << std::left
              << std::setw(16) << L2_error_two << std::endl;  // debug purpose
  }

  // Plot the computed L2error rate
  double eoc_value = eoc(Ndof_array, errors_array);
  std::cout << "EOC value: " << eoc_value << "\n";
}

/**
 * @brief Compare the convergence of multiple methods
 * @tparam SCALAR type of the solution vector
 * @tparam MF type of exact solution mesh function
 * @param solution_wrappers_with_name vector of pairs containing the
 * corresponding solution wrapper datastructure together with the name of the
 * method used in the experimnt (used to label columns of the results)
 * @param exact_solution exact solution mesh function
 * @param experiment_name name to identify the experiment corresponding to the
 * solutions
 * @param isLinear flag for determining if we look at a linear FE space or
 * quadratic FE space - default is quadratic
 */
template <typename SCALAR, typename MF>
void convergence_comparison_multiple_methods(
    std::vector<std::pair<ExperimentSolutionWrapper<SCALAR>, std::string>>
        solution_wrappers_with_name,
    MF& exact_solution, std::string experiment_name, bool isLinear = false) {
  // generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir);  // if necessary

  // create a csv file to store the results for plotting
  std::ofstream L2norm_csv_file;
  L2norm_csv_file.open(concat(main_dir, "/", experiment_name, "_L2error.csv"));
  // Obtain number of methods to be reported
  const int num_methods = solution_wrappers_with_name.size();
  // include metadata about the experiment such as number of refinement levels
  // and the diffusion coefficient used one can use first wrapper to get this
  // metadata, underlying meshes, refinement levels, and diffusion coefficients
  // are the same for each one
  L2norm_csv_file << "Number of refinement levels: "
                  << solution_wrappers_with_name.at(0).first.refinement_levels
                  << "\n";
  L2norm_csv_file << "Diffusion coefficient: "
                  << solution_wrappers_with_name.at(0).first.eps << "\n";
  // Label columns of final csv
  std::string label_for_columns = "No. of dofs,Meshwidth";
  for (int i = 0; i < num_methods; ++i) {
    label_for_columns += "," + solution_wrappers_with_name.at(i).second;
  }
  L2norm_csv_file << label_for_columns << "\n";

  // Obtain mesh hierarchy used, same for each method so take the first
  lf::refinement::MeshHierarchy& multi_mesh{
      *solution_wrappers_with_name.at(0).first.mesh_hierarchy_p};
  std::cout << "Results processing mesh hierarchy info: \n";
  multi_mesh.PrintInfo(std::cout);
  // get number of levels
  auto L = multi_mesh.NumLevels();

  Eigen::VectorXd Ndof_array(L);
  Eigen::VectorXd errors_array(L);

  for (int l = 0; l < L; ++l) {
    // get mesh for refinement level l
    std::shared_ptr<const lf::mesh::Mesh> mesh_l{multi_mesh.getMesh(l)};
    // compute meshwidth
    const double kHMax =
        ecu_scheme::post_processing::ComputeMeshWidthTria(mesh_l);
    // get finite element space for refinement level l
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space;
    if (isLinear) {
      // Linear FE space
      fe_space =
          std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<SCALAR>>(mesh_l);
    } else {
      // Quadratic FE space
      fe_space =
          std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<SCALAR>>(mesh_l);
    }
    // Add results to L2norm_csv_file file
    L2norm_csv_file << fe_space->LocGlobMap().NumDofs() << "," << kHMax;
    for (int i = 0; i < num_methods; ++i) {
      // get solution vector computed at refinement l
      Eigen::VectorXd solution_vec =
          solution_wrappers_with_name.at(i).first.final_time_solutions.at(l);
      // Take finite element solution and wrap it into a mesh function
      auto mf_fe_sol = lf::fe::MeshFunctionFE(fe_space, solution_vec);
      // Compute the L2 error
      double L2_error = std::sqrt(lf::fe::IntegrateMeshFunction(
          *(fe_space->Mesh()),
          lf::mesh::utils::squaredNorm(mf_fe_sol - exact_solution), 10));
      L2norm_csv_file << "," << L2_error;
      if (i == 0) {
        Ndof_array(l) = fe_space->LocGlobMap().NumDofs();
        errors_array(l) = L2_error;
      }
      //      std::cout << std::left << "method " << i << " " << std::setw(10)
      //      << fe_space->LocGlobMap().NumDofs() << std::left <<std::setw(16)
      //      << L2_error << std::endl; //debug purpose
    }
    L2norm_csv_file << "\n";
  }
  // Plot the computed L2error rate
  double eoc_value = eoc(Ndof_array, errors_array);
  std::cout << "EOC value: " << eoc_value << "\n";
}

}  // namespace ecu_scheme::post_processing

#endif  // THESIS_POST_PROCESSING_RESULTS_PROCESSING_H_
