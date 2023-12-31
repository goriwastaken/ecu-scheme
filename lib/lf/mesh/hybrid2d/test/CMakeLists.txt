include(GoogleTest)

set(sources
  mesh_factory_tests.cc
  mesh_factory_test.h
  mesh_orientation_test.cc
)

add_executable(lf.mesh.hybrid2d.test ${sources})
target_link_libraries(lf.mesh.hybrid2d.test PUBLIC Eigen3::Eigen Boost::boost GTest::gtest_main
  lf.io
  lf.mesh.hybrid2d
  lf.mesh.test_utils
  lf.mesh.utils)
set_target_properties(lf.mesh.hybrid2d.test PROPERTIES FOLDER "lib")
target_compile_features(lf.mesh.hybrid2d.test PUBLIC cxx_std_17)
gtest_discover_tests(lf.mesh.hybrid2d.test)

# set(orisources
#   mesh_factory_test.h
#   mesh_orientation_test.cc
# )

# add_executable(lf.mesh.orientation.test ${orisources})
# target_link_libraries(lf.mesh.orientation.test PUBLIC Eigen3::Eigen Boost::boost GTest::gtest_main
#   lf.mesh.hybrid2d
#   lf.mesh.test_utils
#   lf.mesh.utils)
# target_compile_features(lf.mesh.orientation.test PUBLIC cxx_std_17)
# gtest_discover_tests(lf.mesh.orientation.test)
