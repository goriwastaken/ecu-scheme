// Project header file include
#include "manufactured_solution.h"
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

  // Manufactured solution diffusion coefficient
  const double kEps = 1e-10;

  // Manufactured solution velocity field
  const auto kVelocity = [](const Eigen::Vector2d &x){
    return (Eigen::Vector2d() << 2.0, 3.0).finished();
  };
  // Manufactured solution test function f
  const auto kTestFunctor = [kEps](const Eigen::Vector2d &x){
    return 2.0 * (-kEps*x(0) + 3.0*x(0)*x(1) + x(1)*x(1) + (kEps - 3.0*x(1))*exp(2.0*(-1.0 + x(0))/kEps)
        - exp(3.0*(-1.0+x(1))/kEps));
  };
  // Manufactured solution dirichlet function g
  const auto kDirichletFunctor = [kEps](const Eigen::Vector2d &x){
    if(x(0) < 1e-5){
      return -(x(1)*x(1))*exp(-2.0/kEps) + exp(-2.0/kEps + 3.0*(-1.0+x(1))/kEps);
    }else if(x(0) > 1.0 - 1e-5) {
      return 0.0;
    }else if(x(1) < 1e-5){
      return -x(0)*exp(-3.0/kEps) + exp(2.0*(-1.0+x(0))/kEps - 3.0/kEps);
    }else if(x(1) > 1.0 - 1e-5){
      return 0.0;
    }else return 0.0;
  };

  // Wrap all functors into LehrFEM MeshFunctions
  lf::mesh::utils::MeshFunctionConstant mf_eps{-kEps};
  lf::mesh::utils::MeshFunctionGlobal mf_velocity{kVelocity};
  lf::mesh::utils::MeshFunctionGlobal mf_f_test_function{kTestFunctor};
  lf::mesh::utils::MeshFunctionGlobal mf_g_dirichlet{kDirichletFunctor};

  //todo get solution from manufactured solution
  ecu_scheme::experiments::ManufacturedSolutionExperiment<double> experiment(fe_space);
  Eigen::VectorXd solution_vector = experiment.ComputeSolution(kEps, kVelocity, kTestFunctor, kDirichletFunctor);
  //todo process stuff here
  ecu_scheme::post_processing::output_results<double>(fe_space, solution_vector, "manufactured_solution");

  return 0;
}