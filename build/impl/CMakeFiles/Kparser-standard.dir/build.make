# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/daniel/workspace/dynINO

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daniel/workspace/dynINO/build

# Include any dependencies generated for this target.
include impl/CMakeFiles/Kparser-standard.dir/depend.make

# Include the progress variables for this target.
include impl/CMakeFiles/Kparser-standard.dir/progress.make

# Include the compile flags for this target's objects.
include impl/CMakeFiles/Kparser-standard.dir/flags.make

impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o: impl/CMakeFiles/Kparser-standard.dir/flags.make
impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o: ../impl/Kparser-standard.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/daniel/workspace/dynINO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o -c /home/daniel/workspace/dynINO/impl/Kparser-standard.cc

impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.i"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/workspace/dynINO/impl/Kparser-standard.cc > CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.i

impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.s"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/workspace/dynINO/impl/Kparser-standard.cc -o CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.s

impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o.requires:

.PHONY : impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o.requires

impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o.provides: impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o.requires
	$(MAKE) -f impl/CMakeFiles/Kparser-standard.dir/build.make impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o.provides.build
.PHONY : impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o.provides

impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o.provides.build: impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o


impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o: impl/CMakeFiles/Kparser-standard.dir/flags.make
impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o: ../impl/pretrained.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/daniel/workspace/dynINO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Kparser-standard.dir/pretrained.cc.o -c /home/daniel/workspace/dynINO/impl/pretrained.cc

impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Kparser-standard.dir/pretrained.cc.i"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/workspace/dynINO/impl/pretrained.cc > CMakeFiles/Kparser-standard.dir/pretrained.cc.i

impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Kparser-standard.dir/pretrained.cc.s"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/workspace/dynINO/impl/pretrained.cc -o CMakeFiles/Kparser-standard.dir/pretrained.cc.s

impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o.requires:

.PHONY : impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o.requires

impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o.provides: impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o.requires
	$(MAKE) -f impl/CMakeFiles/Kparser-standard.dir/build.make impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o.provides.build
.PHONY : impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o.provides

impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o.provides.build: impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o


impl/CMakeFiles/Kparser-standard.dir/eval.cc.o: impl/CMakeFiles/Kparser-standard.dir/flags.make
impl/CMakeFiles/Kparser-standard.dir/eval.cc.o: ../impl/eval.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/daniel/workspace/dynINO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object impl/CMakeFiles/Kparser-standard.dir/eval.cc.o"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Kparser-standard.dir/eval.cc.o -c /home/daniel/workspace/dynINO/impl/eval.cc

impl/CMakeFiles/Kparser-standard.dir/eval.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Kparser-standard.dir/eval.cc.i"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/workspace/dynINO/impl/eval.cc > CMakeFiles/Kparser-standard.dir/eval.cc.i

impl/CMakeFiles/Kparser-standard.dir/eval.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Kparser-standard.dir/eval.cc.s"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/workspace/dynINO/impl/eval.cc -o CMakeFiles/Kparser-standard.dir/eval.cc.s

impl/CMakeFiles/Kparser-standard.dir/eval.cc.o.requires:

.PHONY : impl/CMakeFiles/Kparser-standard.dir/eval.cc.o.requires

impl/CMakeFiles/Kparser-standard.dir/eval.cc.o.provides: impl/CMakeFiles/Kparser-standard.dir/eval.cc.o.requires
	$(MAKE) -f impl/CMakeFiles/Kparser-standard.dir/build.make impl/CMakeFiles/Kparser-standard.dir/eval.cc.o.provides.build
.PHONY : impl/CMakeFiles/Kparser-standard.dir/eval.cc.o.provides

impl/CMakeFiles/Kparser-standard.dir/eval.cc.o.provides.build: impl/CMakeFiles/Kparser-standard.dir/eval.cc.o


impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o: impl/CMakeFiles/Kparser-standard.dir/flags.make
impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o: ../impl/oracle.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/daniel/workspace/dynINO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Kparser-standard.dir/oracle.cc.o -c /home/daniel/workspace/dynINO/impl/oracle.cc

impl/CMakeFiles/Kparser-standard.dir/oracle.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Kparser-standard.dir/oracle.cc.i"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/workspace/dynINO/impl/oracle.cc > CMakeFiles/Kparser-standard.dir/oracle.cc.i

impl/CMakeFiles/Kparser-standard.dir/oracle.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Kparser-standard.dir/oracle.cc.s"
	cd /home/daniel/workspace/dynINO/build/impl && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/workspace/dynINO/impl/oracle.cc -o CMakeFiles/Kparser-standard.dir/oracle.cc.s

impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o.requires:

.PHONY : impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o.requires

impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o.provides: impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o.requires
	$(MAKE) -f impl/CMakeFiles/Kparser-standard.dir/build.make impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o.provides.build
.PHONY : impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o.provides

impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o.provides.build: impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o


# Object files for target Kparser-standard
Kparser__standard_OBJECTS = \
"CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o" \
"CMakeFiles/Kparser-standard.dir/pretrained.cc.o" \
"CMakeFiles/Kparser-standard.dir/eval.cc.o" \
"CMakeFiles/Kparser-standard.dir/oracle.cc.o"

# External object files for target Kparser-standard
Kparser__standard_EXTERNAL_OBJECTS =

impl/Kparser-standard: impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o
impl/Kparser-standard: impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o
impl/Kparser-standard: impl/CMakeFiles/Kparser-standard.dir/eval.cc.o
impl/Kparser-standard: impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o
impl/Kparser-standard: impl/CMakeFiles/Kparser-standard.dir/build.make
impl/Kparser-standard: cnn/libcnn.a
impl/Kparser-standard: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.58.0
impl/Kparser-standard: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.58.0
impl/Kparser-standard: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.58.0
impl/Kparser-standard: impl/CMakeFiles/Kparser-standard.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/daniel/workspace/dynINO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable Kparser-standard"
	cd /home/daniel/workspace/dynINO/build/impl && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Kparser-standard.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
impl/CMakeFiles/Kparser-standard.dir/build: impl/Kparser-standard

.PHONY : impl/CMakeFiles/Kparser-standard.dir/build

impl/CMakeFiles/Kparser-standard.dir/requires: impl/CMakeFiles/Kparser-standard.dir/Kparser-standard.cc.o.requires
impl/CMakeFiles/Kparser-standard.dir/requires: impl/CMakeFiles/Kparser-standard.dir/pretrained.cc.o.requires
impl/CMakeFiles/Kparser-standard.dir/requires: impl/CMakeFiles/Kparser-standard.dir/eval.cc.o.requires
impl/CMakeFiles/Kparser-standard.dir/requires: impl/CMakeFiles/Kparser-standard.dir/oracle.cc.o.requires

.PHONY : impl/CMakeFiles/Kparser-standard.dir/requires

impl/CMakeFiles/Kparser-standard.dir/clean:
	cd /home/daniel/workspace/dynINO/build/impl && $(CMAKE_COMMAND) -P CMakeFiles/Kparser-standard.dir/cmake_clean.cmake
.PHONY : impl/CMakeFiles/Kparser-standard.dir/clean

impl/CMakeFiles/Kparser-standard.dir/depend:
	cd /home/daniel/workspace/dynINO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daniel/workspace/dynINO /home/daniel/workspace/dynINO/impl /home/daniel/workspace/dynINO/build /home/daniel/workspace/dynINO/build/impl /home/daniel/workspace/dynINO/build/impl/CMakeFiles/Kparser-standard.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : impl/CMakeFiles/Kparser-standard.dir/depend

