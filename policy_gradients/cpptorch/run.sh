mkdir -p build
cd build
cmake -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_PREFIX_PATH=/Users/adamkowalski/pytorch/torch ..
cmake --build .
mv compile_commands.json ..
./cpptorch
