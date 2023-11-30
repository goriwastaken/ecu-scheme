project(projects.ecu_scheme.experiments.testexp)

#set(SOURCES
#        expfittedupwind.cc
#        expfittedupwind.h
#        expfittedupwind_main.cc)
#set(LIBRARIES
#        Eigen3::Eigen
#        LF::lf.assemble
#        LF::lf.base
#        LF::lf.io
#        LF::lf.mesh.hybrid2d
#        LF::lf.mesh.utils
#        LF::lf.uscalfe
#        LF::lf.refinement)

set(SOURCES
        upwindquadrature_main.cc
        upwindquadrature.h
        upwindquadrature.cc
        )

set(LIBRARIES
        Eigen3::Eigen
        lf.assemble
        lf.base
        lf.geometry
        lf.io
        lf.mesh.hybrid2d
        lf.mesh
        lf.mesh.utils
        lf.uscalfe
        )


set(TARGET ${PROJECT_NAME}.exp1)

add_executable(${TARGET} ${SOURCES})
# set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME ${OUTPUT_NAME})
target_compile_definitions(${TARGET} PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}\")
target_compile_definitions(${TARGET} PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
# For including all symbols in the executable: Does not work on Max OS X
# target_link_libraries(${TARGET} PUBLIC   "-Wl,--whole-archive" ${LIBRARIES} "-Wl,--no-whole-archive")
target_link_libraries(${TARGET} PUBLIC ${LIBRARIES})
set_target_properties(${TARGET} PROPERTIES FOLDER "projects/ecu_scheme")
target_compile_features(${TARGET} PUBLIC cxx_std_17)

add_library(${TARGET}.static STATIC ${SOURCES})
target_compile_definitions(${TARGET}.static PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}\")
target_compile_definitions(${TARGET}.static PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
target_link_libraries(${TARGET}.static PUBLIC ${LIBRARIES})
set_target_properties(${TARGET}.static PROPERTIES FOLDER "projects/ecu_scheme")
target_include_directories(${TARGET}.static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Include the tests
add_subdirectory(test)