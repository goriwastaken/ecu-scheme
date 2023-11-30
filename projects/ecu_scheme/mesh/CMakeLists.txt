set(SOURCES
        mesh.h
        basic_mesh_builder.cc basic_mesh_builder.h)
set(LIBRARIES
        Eigen3::Eigen
        lf.base
        lf.geometry
        lf.io
        lf.mesh
        lf.mesh.utils
        lf.mesh.hybrid2d
        Boost::program_options
        )

add_library(${PROJECT_NAME}.mesh ${SOURCES})
target_link_libraries(${PROJECT_NAME}.mesh PUBLIC ${LIBRARIES})
set_target_properties(${TARGET} PROPERTIES FOLDER "projects/ecu_scheme")
target_include_directories(${PROJECT_NAME}.mesh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${PROJECT_NAME}.mesh PUBLIC cxx_std_17)

# Include the tests
#add_subdirectory(test)
