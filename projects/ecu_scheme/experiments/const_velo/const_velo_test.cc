// Project header file include
// C system headers
// C++ standard library headers
#include <iostream>
#include <cmath>
// Other libraries headers
#include "lf/uscalfe/uscalfe.h"
#include "mesh.h"

/**
 * @brief Test of Convergence: Constant Velocity experiment from the dissertation Cecilia Pagliantini. Computational Magnetohydrodynamics with Dis-
crete Differential Forms. PhD thesis, ETH Zürich, 2016. Chapter 4.3.1
 */
int main(int argc, char* argv[]){
  if(argc != 0 && argv != nullptr){
    std::cerr << "Usage todo " << std::endl;
  }
  // setup is [0,2]x[0,2] with periodic boundary conditions and time interval I= [0,0.5]
  // initial condition A_0 = 1/pi * cos(pi*y) + 1/(2pi) * cos(2pi*x)
  // velocity field u = (4,4)
  // forcing term f = 0
  // use Heun time-stepping with uniform time step delta_t = 0.1*h for bilinear Lagrangian FE and delta_t = 0.01*h for biquadratic Lagrangian FE
  // compare solution at final time t=0.5 with initial condition
  // report L2 and H1 error norms

  ecu_scheme::mesh::BasicMeshBuilder builder;
  std::shared_ptr<lf::mesh::Mesh> mesh_p = builder.Build(0.0, 0.0, 2.0, 2.0);

  auto fe_space_quad = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
  auto fe_space_linear = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);

  const double delta_t_linear = 0.1;
  const double delta_t_quad = 0.01;

  // initial condition for the experiment
  const auto kAInitCondition = [](const Eigen::Vector2d &x){
    return 1.0/M_PI * std::cos(M_PI*x(1)) + 1.0/(2.0*M_PI) * std::cos(2.0*M_PI*x(0));
  };
  // constant velocity field
  const auto kVelocity = [](const Eigen::Vector2d &x){
    return (Eigen::Vector2d() << 4.0, 4.0).finished();
  };
  // forcing term f
  const auto kTestFunctor = [](const Eigen::Vector2d &x){
    return 0.0;
  };
  // todo implement periodic boundary conditions?

  // Wrap all functors into LehrFEM MeshFunctions
  lf::mesh::utils::MeshFunctionGlobal mf_a_init{kAInitCondition};
  lf::mesh::utils::MeshFunctionGlobal mf_velocity{kVelocity};
  // we just consider the pure advection case
  //lf::mesh::utils::MeshFunctionGlobal mf_f_test_function{kTestFunctor};

  //todo ecu_scheme::experiments::ConstantVelocitySolution<double> experiment(fe_space_quadratic/linear);
  //Eigen::VectorXd solution_vector = Eigen::VectorXd::Zero(1);

  return 0;
}