
include(GoogleTest)

set(sources
  gmsh_file_v4_tests.cc
  gmsh_reader_tests.cc
  vtk_writer_tests.cc
)

add_executable(lf.io.test ${sources})
target_link_libraries(lf.io.test PUBLIC Eigen3::Eigen Boost::boost GTest::gtest_main lf.io lf.io.test_utils lf.mesh.hybrid2d lf.mesh.test_utils lf.quad lf.refinement)
set_target_properties(lf.io.test PROPERTIES FOLDER "lib")
gtest_discover_tests(lf.io.test)
