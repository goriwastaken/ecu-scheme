# cmake configuration file for compilation of LehrFEM++ examples: RefEl

set(runtime_test runtime_test.cc)

add_executable(experiments.efficiency.runtime_test ${runtime_test})

target_link_libraries(experiments.efficiency.runtime_test
  PUBLIC Eigen3::Eigen Boost::boost lf.base)

set_target_properties(experiments.efficiency.runtime_test PROPERTIES FOLDER "experiments")

target_compile_features(experiments.efficiency.runtime_test PUBLIC cxx_std_17)

  
    

  

