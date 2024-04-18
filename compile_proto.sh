protoc --cpp_out=./foxglove_pb ./foxglove/*.proto
rm -r ./include/foxglove
rm ./foxglove_pb/*.cc
mv foxglove_pb/foxglove/ include/
mv include/foxglove/*.cc foxglove_pb/
