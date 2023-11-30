include(GoogleTest)

set(SOURCES
        upwindquadrature_test.cc
        )

set(LIBRARIES
        Eigen3::Eigen
        GTest::gtest_main
        lf.geometry
        lf.mesh.hybrid2d
        lf.io
        lf.mesh
        ${PROJECT_NAME}.exp1.static
        )

set(TARGET ${PROJECT_NAME}.test)

#get name of parent directory of "test"
#get_filename_component(TARGET_TO_TEST "${PROJECT_NAME}.${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

add_executable(${TARGET} ${SOURCES})
target_compile_definitions(${TARGET} PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/test\")
target_compile_definitions(${TARGET} PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
target_link_libraries(${TARGET} PUBLIC ${LIBRARIES}) #${TARGET_TO_TEST}.static)
target_compile_features(${TARGET} PUBLIC cxx_std_17)
gtest_discover_tests(${TARGET})