set(sources
  hybrid2d.h
  mesh.cc
  mesh.h
  mesh_factory.cc
  mesh_factory.h
  point.h
  point.cc
  quad.h
  quad.cc
  segment.h
  segment.cc
  triangle.h
  triangle.cc
)

lf_add_library(lf.mesh.hybrid2d ${sources})
target_link_libraries(lf.mesh.hybrid2d PUBLIC Eigen3::Eigen lf.base lf.mesh lf.mesh.utils)
set_target_properties(lf.mesh.hybrid2d PROPERTIES FOLDER "lib")

if(LF_ENABLE_TESTING)
  add_subdirectory(test)
endif()
