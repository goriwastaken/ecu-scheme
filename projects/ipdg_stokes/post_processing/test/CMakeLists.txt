include(GoogleTest)

set(sources mesh_function_velocity_tests.cc)

add_executable(projects.ipdg_stokes.post_processing.test ${sources})
target_link_libraries(projects.ipdg_stokes.post_processing.test PUBLIC
		      Eigen3::Eigen GTest::gtest_main lf.base lf.assemble lf.mesh lf.mesh.hybrid2d lf.mesh.utils lf.uscalfe lf.refinement projects.ipdg_stokes.post_processing)
set_target_properties(projects.ipdg_stokes.post_processing.test PROPERTIES FOLDER "projects/ipdg_stokes")
target_compile_features(projects.ipdg_stokes.post_processing.test PUBLIC cxx_std_17)
gtest_discover_tests(projects.ipdg_stokes.post_processing.test)
