cmake_minimum_required(VERSION 3.0.2)
project(mtc_retract_approach)

find_package(catkin REQUIRED COMPONENTS
  moveit_core
  moveit_ros_planning_interface
  moveit_task_constructor_core
  roscpp
)

find_package(fmt REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(tams_pa10_demo src/tams_pa10_demo.cpp)
add_dependencies(tams_pa10_demo ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tams_pa10_demo ${catkin_LIBRARIES} fmt::fmt)