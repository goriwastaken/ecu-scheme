set(SOURCES
        assemble.h
        diffusion_matrix_provider.h
        diffusion_matrix_provider.cc
        convection_upwind_matrix_provider.h
        convection_upwind_matrix_provider.cc
        load_vector_provider.h supg_matrix_provider.cc supg_matrix_provider.h lumped_mass_matrix_provider.cc lumped_mass_matrix_provider.h supg_linear_matrix_provider.h supg_linear_matrix_provider.cc)
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
