# cmake configuration file for compilation of LehrFEM++ examples: RefEl

set(quad_demo quad_demo.cc)

add_executable(examples.quad.quad_demo ${quad_demo})

target_link_libraries(examples.quad.quad_demo
  PUBLIC Eigen3::Eigen Boost::program_options
  lf.mesh
  lf.mesh.test_utils
  lf.mesh.utils
  lf.mesh.hybrid2d
  lf.refinement
  lf.io
  lf.quad
  )
set_target_properties(examples.quad.quad_demo PROPERTIES FOLDER "examples") 

if(MSVC)
  target_compile_options(examples.quad.quad_demo PRIVATE /bigobj)
endif()

add_custom_target(examples.quad.quad_demo_run COMMAND examples.quad.quad_demo)
set_target_properties(examples.quad.quad_demo_run PROPERTIES FOLDER "examples")
add_dependencies(examples_run examples.quad.quad_demo_run)
    

  

