set(SOURCES
        assemble.h
        convection_upwind_matrix_provider.h
        convection_upwind_matrix_provider.cc
        supg_matrix_provider.cc supg_matrix_provider.h supg_linear_matrix_provider.h supg_linear_matrix_provider.cc mesh_function_one_form.h stable_convection_prov.cc stable_convection_prov.h fifteen_point_upwind_matrix_provider.cc fifteen_point_upwind_matrix_provider.h edge_element_mass_matrix_provider.cc edge_element_mass_matrix_provider.h edge_element_grad_matrix_provider.cc edge_element_grad_matrix_provider.h)
set(LIBRARIES
        Eigen3::Eigen
        lf.assemble
        lf.base
        lf.geometry
        lf.io
        lf.mesh
        lf.mesh.utils
        lf.uscalfe
        #todo add thesis mesh library probably
        )

add_library(${PROJECT_NAME}.assemble ${SOURCES})
target_link_libraries(${PROJECT_NAME}.assemble PUBLIC ${LIBRARIES})
set_target_properties(${TARGET} PROPERTIES FOLDER "projects/ecu_scheme")
target_include_directories(${PROJECT_NAME}.assemble PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${PROJECT_NAME}.assemble PUBLIC cxx_std_17)

# Include the tests
#add_subdirectory(test)
