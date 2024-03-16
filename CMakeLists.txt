cmake_minimum_required(VERSION 3.15)
project(FoxgloveWebSocket CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LIB_PATH "/usr/local/lib") 
set(INCLUDE_PATH "/usr/local/include")
set(COMMON_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

include_directories(${COMMON_INCLUDE_DIR} ${INCLUDE_PATH})

find_package(OpenSSL REQUIRED)

add_subdirectory(foxglove_pb)

add_executable(server_protobuf src/server_protobuf.cpp)
target_include_directories(server_protobuf PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(server_protobuf PRIVATE ${LIB_PATH}/libfoxglove_websocket.a)
target_link_libraries(server_protobuf PRIVATE foxglove_pb)
target_link_libraries(server_protobuf PRIVATE protobuf)
target_link_libraries(server_protobuf PRIVATE pthread)
target_link_libraries(server_protobuf PRIVATE OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(server_protobuf PRIVATE z)
