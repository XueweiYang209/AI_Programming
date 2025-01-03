# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yang/aaa/AI_programming

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yang/aaa/AI_programming/build

# Include any dependencies generated for this target.
include CMakeFiles/MyTensor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/MyTensor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/MyTensor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MyTensor.dir/flags.make

CMakeFiles/MyTensor.dir/binding.cpp.o: CMakeFiles/MyTensor.dir/flags.make
CMakeFiles/MyTensor.dir/binding.cpp.o: ../binding.cpp
CMakeFiles/MyTensor.dir/binding.cpp.o: CMakeFiles/MyTensor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yang/aaa/AI_programming/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MyTensor.dir/binding.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MyTensor.dir/binding.cpp.o -MF CMakeFiles/MyTensor.dir/binding.cpp.o.d -o CMakeFiles/MyTensor.dir/binding.cpp.o -c /home/yang/aaa/AI_programming/binding.cpp

CMakeFiles/MyTensor.dir/binding.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MyTensor.dir/binding.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yang/aaa/AI_programming/binding.cpp > CMakeFiles/MyTensor.dir/binding.cpp.i

CMakeFiles/MyTensor.dir/binding.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MyTensor.dir/binding.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yang/aaa/AI_programming/binding.cpp -o CMakeFiles/MyTensor.dir/binding.cpp.s

CMakeFiles/MyTensor.dir/Tensor.cu.o: CMakeFiles/MyTensor.dir/flags.make
CMakeFiles/MyTensor.dir/Tensor.cu.o: ../Tensor.cu
CMakeFiles/MyTensor.dir/Tensor.cu.o: CMakeFiles/MyTensor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yang/aaa/AI_programming/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/MyTensor.dir/Tensor.cu.o"
	/home/yang/.micromamba/envs/py/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/MyTensor.dir/Tensor.cu.o -MF CMakeFiles/MyTensor.dir/Tensor.cu.o.d -x cu -dc /home/yang/aaa/AI_programming/Tensor.cu -o CMakeFiles/MyTensor.dir/Tensor.cu.o

CMakeFiles/MyTensor.dir/Tensor.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/MyTensor.dir/Tensor.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/MyTensor.dir/Tensor.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/MyTensor.dir/Tensor.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/MyTensor.dir/Module.cu.o: CMakeFiles/MyTensor.dir/flags.make
CMakeFiles/MyTensor.dir/Module.cu.o: ../Module.cu
CMakeFiles/MyTensor.dir/Module.cu.o: CMakeFiles/MyTensor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yang/aaa/AI_programming/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/MyTensor.dir/Module.cu.o"
	/home/yang/.micromamba/envs/py/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/MyTensor.dir/Module.cu.o -MF CMakeFiles/MyTensor.dir/Module.cu.o.d -x cu -dc /home/yang/aaa/AI_programming/Module.cu -o CMakeFiles/MyTensor.dir/Module.cu.o

CMakeFiles/MyTensor.dir/Module.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/MyTensor.dir/Module.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/MyTensor.dir/Module.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/MyTensor.dir/Module.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target MyTensor
MyTensor_OBJECTS = \
"CMakeFiles/MyTensor.dir/binding.cpp.o" \
"CMakeFiles/MyTensor.dir/Tensor.cu.o" \
"CMakeFiles/MyTensor.dir/Module.cu.o"

# External object files for target MyTensor
MyTensor_EXTERNAL_OBJECTS =

CMakeFiles/MyTensor.dir/cmake_device_link.o: CMakeFiles/MyTensor.dir/binding.cpp.o
CMakeFiles/MyTensor.dir/cmake_device_link.o: CMakeFiles/MyTensor.dir/Tensor.cu.o
CMakeFiles/MyTensor.dir/cmake_device_link.o: CMakeFiles/MyTensor.dir/Module.cu.o
CMakeFiles/MyTensor.dir/cmake_device_link.o: CMakeFiles/MyTensor.dir/build.make
CMakeFiles/MyTensor.dir/cmake_device_link.o: /home/yang/.micromamba/envs/py/lib/libcublas.so
CMakeFiles/MyTensor.dir/cmake_device_link.o: /home/yang/.micromamba/envs/py/lib/libcudart.so
CMakeFiles/MyTensor.dir/cmake_device_link.o: CMakeFiles/MyTensor.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yang/aaa/AI_programming/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CUDA device code CMakeFiles/MyTensor.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MyTensor.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MyTensor.dir/build: CMakeFiles/MyTensor.dir/cmake_device_link.o
.PHONY : CMakeFiles/MyTensor.dir/build

# Object files for target MyTensor
MyTensor_OBJECTS = \
"CMakeFiles/MyTensor.dir/binding.cpp.o" \
"CMakeFiles/MyTensor.dir/Tensor.cu.o" \
"CMakeFiles/MyTensor.dir/Module.cu.o"

# External object files for target MyTensor
MyTensor_EXTERNAL_OBJECTS =

MyTensor.cpython-310-x86_64-linux-gnu.so: CMakeFiles/MyTensor.dir/binding.cpp.o
MyTensor.cpython-310-x86_64-linux-gnu.so: CMakeFiles/MyTensor.dir/Tensor.cu.o
MyTensor.cpython-310-x86_64-linux-gnu.so: CMakeFiles/MyTensor.dir/Module.cu.o
MyTensor.cpython-310-x86_64-linux-gnu.so: CMakeFiles/MyTensor.dir/build.make
MyTensor.cpython-310-x86_64-linux-gnu.so: /home/yang/.micromamba/envs/py/lib/libcublas.so
MyTensor.cpython-310-x86_64-linux-gnu.so: /home/yang/.micromamba/envs/py/lib/libcudart.so
MyTensor.cpython-310-x86_64-linux-gnu.so: CMakeFiles/MyTensor.dir/cmake_device_link.o
MyTensor.cpython-310-x86_64-linux-gnu.so: CMakeFiles/MyTensor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yang/aaa/AI_programming/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared module MyTensor.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MyTensor.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/yang/aaa/AI_programming/build/MyTensor.cpython-310-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/MyTensor.dir/build: MyTensor.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/MyTensor.dir/build

CMakeFiles/MyTensor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MyTensor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MyTensor.dir/clean

CMakeFiles/MyTensor.dir/depend:
	cd /home/yang/aaa/AI_programming/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yang/aaa/AI_programming /home/yang/aaa/AI_programming /home/yang/aaa/AI_programming/build /home/yang/aaa/AI_programming/build /home/yang/aaa/AI_programming/build/CMakeFiles/MyTensor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MyTensor.dir/depend
