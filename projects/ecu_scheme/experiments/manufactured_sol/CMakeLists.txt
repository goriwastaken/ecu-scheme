project(projects.ecu_scheme.experiments.manufactured_solution)

set(SOURCES
        manufactured_solution.cc
        manufactured_solution.h
        manufactured_test.cc
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

#todo add custom target to run experiment separately with specific command
#add_custom_target(${TARGET}.run
#        COMMAND ${TARGET}
#        DEPENDS ${TARGET}
#        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#        COMMENT "Running ${TARGET}"
#        VERBATIM
#        )