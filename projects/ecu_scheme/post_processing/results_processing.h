
#ifndef THESIS_POST_PROCESSING_RESULTS_PROCESSING_H_
#define THESIS_POST_PROCESSING_RESULTS_PROCESSING_H_

#include <lf/fe/fe.h>
#include <lf/uscalfe/uscalfe.h>
#include <lf/io/io.h>

#include <string>
#include <iomanip>
#include <filesystem>
#include <utility>
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

//todo augment method to also create convergence results
template<typename SCALAR>
void output_results(std::shared_ptr<const lf::fe::ScalarFESpace<SCALAR>> fe_space,
                    Eigen::VectorXd solution_vec,
                    std::string experiment_name){

  //generate location for output
  std::string main_dir = PROJECT_BUILD_DIR "/results";
  std::filesystem::create_directory(main_dir);

  //output results to vtk file
  lf::fe::MeshFunctionFE mf_sol(fe_space, std::move(solution_vec));
  lf::io::VtkWriter vtk_writer(fe_space->Mesh(), concat(main_dir, "/", experiment_name, ".vtk"));
  vtk_writer.WritePointData(concat(experiment_name, "_solution"), mf_sol);
}

} // post_processing

#endif //THESIS_POST_PROCESSING_RESULTS_PROCESSING_H_
