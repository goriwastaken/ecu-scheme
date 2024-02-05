set(SOURCES
        post_processing.h
        results_processing.h norms.h ecu_tools.cc ecu_tools.h)
set(LIBRARIES
        Eigen3::Eigen
        lf.base
        lf.geometry
        lf.io
        lf.uscalfe
        lf.fe
        lf.mesh
        lf.mesh.utils
        lf.mesh.hybrid2d
        lf.assemble
        projects.ecu_scheme.mesh
        )

set(TARGET ${PROJECT_NAME}.postprocessing)

add_library(${TARGET} ${SOURCES})

# We need this variable definition to always generate the results folder in the project build directory
target_compile_definitions(${TARGET} PUBLIC PROJECT_BUILD_DIR=\"${CMAKE_BINARY_DIR}\")

target_link_libraries(${TARGET} PUBLIC ${LIBRARIES})
set_target_properties(${TARGET} PROPERTIES FOLDER "projects/ecu_scheme")
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

# Include the tests
#add_subdirectory(test)