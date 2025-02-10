cmake_minimum_required(VERSION 3.4...3.18)
project(planner)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(pybind11)
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

pybind11_add_module(planner 
  src/main.cpp
  src/grid_map.cpp
  src/penal_traj_opt.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(grid_map
#                            PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(
  planner
  PRIVATE pybind11::module
  ${OpenCV_LIBS}
  ${pybind11_LIBRARIES}
)