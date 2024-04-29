project(projects.ecu_scheme.experiments.rotating_hump)

set(SOURCES
        rotating_hump_test.cc
        rotating_hump_sol.cc
        rotating_hump_sol.h
        )

set(LIBRARIES
        Eigen3::Eigen
        lf.base
        lf.io
        lf.mesh
        lf.mesh.hybrid2d
        lf.mesh.utils
        lf.uscalfe
        lf.refinement
        projects.ecu_scheme.mesh
        projects.ecu_scheme.assemble
        projects.ecu_scheme.postprocessing
        )

set(TARGET ${PROJECT_NAME})

add_executable(${TARGET} ${SOURCES} )
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