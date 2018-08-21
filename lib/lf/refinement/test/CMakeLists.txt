include(GoogleTest)

set(sources
  refinement_test_utils.h
  refinement_test_utils.cc
    father_child_relation_tests.cc 
    geo_ref_test.cc
    hybrid2d_refinement_pattern_tests.cc
    regreftest.cc
)

add_executable(lf.refinement.test ${sources})
target_link_libraries(lf.refinement.test
  PUBLIC Eigen3::Eigen Boost::boost GTest::main
  lf.refinement
  lf.mesh.hybrid2dp
  lf.mesh.test_utils
  lf.mesh.utils
  lf.io
  lf.io.test_utils
  )
target_compile_features(lf.refinement.test PUBLIC cxx_std_17)
gtest_discover_tests(lf.refinement.test)

set(rrt_sources 
  refinement_test_utils.h
  refinement_test_utils.cc
  regreftest.cc
)

add_executable(lf.refinement.rrt_test ${rrt_sources})
target_link_libraries(lf.refinement.rrt_test
  PUBLIC Eigen3::Eigen Boost::boost GTest::main
  lf.refinement
  lf.mesh.hybrid2dp
  lf.mesh.test_utils
  lf.mesh.utils
  lf.io
  lf.io.test_utils
  )
target_compile_features(lf.refinement.rrt_test PUBLIC cxx_std_17)
gtest_discover_tests(lf.refinement.rrt_test)