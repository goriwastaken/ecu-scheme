set(sources
  refinement.h refinement.cc
  hybrid2d_refinement_pattern.h hybrid2d_refinement_pattern.cc
  mesh_hierarchy.h mesh_hierarchy.cc
  refutils.h refutils.cc
  mesh_function_transfer.h
  )

lf_add_library(lf.refinement ${sources})
target_link_libraries(lf.refinement PUBLIC Eigen3::Eigen lf.base lf.geometry lf.io lf.fe lf.mesh.hybrid2d)
set_target_properties(lf.refinement PROPERTIES FOLDER "lib")

if(LF_ENABLE_TESTING)
  add_subdirectory(test)
endif()
