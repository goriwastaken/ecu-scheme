// Project header file include
#include "concentric_stream_solution.h"
// C system headers
// C++ standard library headers
#include <cmath>
// Other libraries headers
#include "lf/uscalfe/uscalfe.h"
#include "mesh.h"
#include "post_processing.h"

int main(int argc, char* argv[]){
  if(argc != 0 && argv != nullptr){
    std::cerr << "Usage todo " << std::endl;
  }

  ecu_scheme::mesh::BasicMeshBuilder builder;
  std::shared_ptr<lf::mesh::Mesh> mesh_p = builder.Build();

  auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);

//  const lf::assemble::DofHandler &dofh{fe_space->LocGlobMap()};

  // Concentric circles diffusion coefficient
  const double kEps = 1e-4;

  // Concentric circle streamline velocity field
  const auto kVelocity = [](const Eigen::Vector2d &x){
    return (Eigen::Vector2d() << -x(1), x(0)).finished();
  };

  // Concentric circle streamlines dirichlet function g
  const auto kDirichletFunctor = [](const Eigen::Vector2d &x){
    return x(1) == 0 ? 0.5 - std::abs(x(0) - 0.5) : 0.0;
  };

  // Wrap all functors into LehrFEM MeshFunctions
  lf::mesh::utils::MeshFunctionConstant mf_eps{-kEps};
  lf::mesh::utils::MeshFunctionGlobal mf_velocity{kVelocity};
  //lf::mesh::utils::MeshFunctionGlobal mf_f_test_function{kTestFunctor};
  lf::mesh::utils::MeshFunctionGlobal mf_g_dirichlet{kDirichletFunctor};

  //todo get solution from manufactured solution
  ecu_scheme::experiments::ConcentricStreamSolution<double> experiment(fe_space);
  Eigen::VectorXd solution_vector = experiment.ComputeSolution(kEps, kVelocity, kDirichletFunctor);
  //todo process stuff here
  ecu_scheme::post_processing::output_results<double>(fe_space, solution_vector, "concentric_stream_solution");


  return 0;
}