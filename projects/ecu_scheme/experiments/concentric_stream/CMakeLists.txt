project(projects.ecu_scheme.experiments.concentric_stream)

set(SOURCES
        concentric_stream_solution.h
        concentric_stream_test.cc
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

add_executable(${TARGET} ${SOURCES} )
target_compile_definitions(${TARGET} PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}\")
target_compile_definitions(${TARGET} PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
target_link_libraries(${TARGET} ${LIBRARIES})

add_library(${TARGET}.static STATIC ${SOURCES})
target_compile_definitions(${TARGET}.static PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}\")
target_compile_definitions(${TARGET}.static PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
target_link_libraries(${TARGET}.static PUBLIC ${LIBRARIES})
target_include_directories(${TARGET}.static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})