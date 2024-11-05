cmake_minimum_required(VERSION 3.15)
project(FoxgloveWebSocket C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenSSL REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(thirdparty/foxglove-websocket)
add_subdirectory(foxglove_pb)

include_directories(include ${INCLUDE_PATH} ${foxglove_websocket_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(server_protobuf src/server_protobuf.cpp)
target_include_directories(server_protobuf PRIVATE 
                            ${PROJECT_SOURCE_DIR}/include
                            foxglove_websocket
                            foxglove_pb
                            protobuf
                            pthread
                            OpenSSL::SSL OpenSSL::Crypto
                            z
                      )
target_link_libraries(server_protobuf PRIVATE 
                            foxglove_websocket
                            foxglove_pb
                            protobuf
                            pthread
                            OpenSSL::SSL OpenSSL::Crypto
                            z
                            ${OpenCV_LIBS} ${PROTOBUF_LIBRARY}
                      )

add_executable(rm_robot_sim src/rm_robot_sim.cpp include/robot/robot_armor.hpp include/robot/camera.hpp)
target_link_libraries(rm_robot_sim PRIVATE 
                            foxglove_websocket
                            foxglove_pb
                            protobuf
                            pthread
                            OpenSSL::SSL OpenSSL::Crypto
                            z
                            ${OpenCV_LIBS} ${PROTOBUF_LIBRARY}
                      )