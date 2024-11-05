# foxglove_cpp_example

![](./doc/录制_2024_04_09_19_44_15_553.gif)

依赖于[foxglove/ws-protocol/cpp/foxglove-websocket](https://github.com/foxglove/ws-protocol/tree/main/cpp/foxglove-websocket)

## Build and Run
```
sudo apt install nlohmann-json3-dev
sudo apt install libwebsocketpp-dev
sudo apt install libprotobuf-dev
./compile_proto.sh
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build -j
```

