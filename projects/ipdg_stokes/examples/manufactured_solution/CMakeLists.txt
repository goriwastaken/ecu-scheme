project(projects.ipdg_stokes.examples.manufactured_solution)

add_executable(${PROJECT_NAME}.main manufactured_solution.cc)
target_link_libraries(${PROJECT_NAME}.main PUBLIC Eigen3::Eigen lf.base lf.mesh lf.mesh.hybrid2d lf.quad lf.refinement lf.io projects.ipdg_stokes.assemble projects.ipdg_stokes.mesh projects.ipdg_stokes.post_processing)
set_target_properties(${PROJECT_NAME}.main PROPERTIES FOLDER "projects/ipdg_stokes")
target_compile_features(${PROJECT_NAME}.main PUBLIC cxx_std_17)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output.dat
		   DEPENDS ${PROJECT_NAME}.main
		   COMMAND $<TARGET_FILE:${PROJECT_NAME}.main> > output.dat
		   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(${PROJECT_NAME}.plot
		  COMMAND gnuplot ${CMAKE_CURRENT_SOURCE_DIR}/plot.gnu
		  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output.dat ${CMAKE_CURRENT_SOURCE_DIR}/plot.gnu
		  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_dependencies(projects.ipdg_stokes.plots ${PROJECT_NAME}.plot)

set_target_properties(${PROJECT_NAME}.plot PROPERTIES FOLDER "projects/ipdg_stokes")