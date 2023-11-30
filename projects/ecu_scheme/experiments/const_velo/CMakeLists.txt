project(projects.ecu_scheme.experiments.const_velo)

set(SOURCES
        const_velo_test.cc
        )

set(LIBRARIES
        Eigen3::Eigen
        lf.base
        lf.io
        lf.mesh
        lf.mesh.hybrid2d
        projects.ecu_scheme.mesh
        projects.ecu_scheme.assemble
        projects.ecu_scheme.postprocessing
        )

set(TARGET ${PROJECT_NAME})

add_executable(${TARGET} ${SOURCES} const_velo_solution.cc const_velo_solution.h)
target_compile_definitions(${TARGET} PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}\")
target_compile_definitions(${TARGET} PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
target_link_libraries(${TARGET} ${LIBRARIES})
set_target_properties(${TARGET} PROPERTIES FOLDER "projects/ecu_scheme")

add_library(${TARGET}.static STATIC ${SOURCES})
target_compile_definitions(${TARGET}.static PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}\")
target_compile_definitions(${TARGET}.static PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
target_link_libraries(${TARGET}.static PUBLIC ${LIBRARIES})
set_target_properties(${TARGET}.static PROPERTIES FOLDER "projects/ecu_scheme")
target_include_directories(${TARGET}.static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})