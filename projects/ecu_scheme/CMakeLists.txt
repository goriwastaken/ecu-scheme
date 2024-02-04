cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

#include("cmake/HunterGate.cmake")
#HunterGate(
#        URL "https://github.com/cpp-pm/hunter/archive/v0.23.321.tar.gz"
#        SHA1 "5e53cbb0429037ea8e2592bfd92704b8ff3ab492"
#)

project(projects.ecu_scheme)
add_custom_target(projects.ecu_scheme.plots)

# add custom targets
#include(${CMAKE_SOURCE_DIR}/cmake/modules/custom_targets.cmake)
#get_custom_targets()
#add_custom_target(plots)

#hunter_add_package(Eigen)
#find_package(Eigen3 CONFIG REQUIRED)
#
#hunter_add_package(GTest)
#find_package(GTest CONFIG REQUIRED)
#
#hunter_add_package(Boost COMPONENTS chrono timer system filesystem program_options)
#find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
#
#hunter_add_package(lehrfempp)
#find_package(lehrfempp CONFIG REQUIRED)


#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#
##Include the GTest directories
#include_directories(${GTEST_INCLUDE_DIRS})
#
##Enable testing
#enable_testing()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,leak,undefined")

#debug show what the source/binary directories are
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

#Add the different subprojects
add_subdirectory(assemble)
add_subdirectory(mesh)
add_subdirectory(post_processing)
add_subdirectory(experiments)