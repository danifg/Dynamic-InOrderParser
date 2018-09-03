# Install script for directory: /home/daniel/workspace/dynINO/cnn

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/cnn" TYPE FILE FILES
    "/home/daniel/workspace/dynINO/cnn/aligned-mem-pool.h"
    "/home/daniel/workspace/dynINO/cnn/cfsm-builder.h"
    "/home/daniel/workspace/dynINO/cnn/c2w.h"
    "/home/daniel/workspace/dynINO/cnn/cnn.h"
    "/home/daniel/workspace/dynINO/cnn/conv.h"
    "/home/daniel/workspace/dynINO/cnn/cuda.h"
    "/home/daniel/workspace/dynINO/cnn/devices.h"
    "/home/daniel/workspace/dynINO/cnn/dict.h"
    "/home/daniel/workspace/dynINO/cnn/dim.h"
    "/home/daniel/workspace/dynINO/cnn/exec.h"
    "/home/daniel/workspace/dynINO/cnn/expr.h"
    "/home/daniel/workspace/dynINO/cnn/fast-lstm.h"
    "/home/daniel/workspace/dynINO/cnn/functors.h"
    "/home/daniel/workspace/dynINO/cnn/gpu-kernels.h"
    "/home/daniel/workspace/dynINO/cnn/gpu-ops.h"
    "/home/daniel/workspace/dynINO/cnn/graph.h"
    "/home/daniel/workspace/dynINO/cnn/gru.h"
    "/home/daniel/workspace/dynINO/cnn/hsm-builder.h"
    "/home/daniel/workspace/dynINO/cnn/init.h"
    "/home/daniel/workspace/dynINO/cnn/lstm.h"
    "/home/daniel/workspace/dynINO/cnn/mem.h"
    "/home/daniel/workspace/dynINO/cnn/model.h"
    "/home/daniel/workspace/dynINO/cnn/mp.h"
    "/home/daniel/workspace/dynINO/cnn/nodes.h"
    "/home/daniel/workspace/dynINO/cnn/param-nodes.h"
    "/home/daniel/workspace/dynINO/cnn/random.h"
    "/home/daniel/workspace/dynINO/cnn/rnn-state-machine.h"
    "/home/daniel/workspace/dynINO/cnn/rnn.h"
    "/home/daniel/workspace/dynINO/cnn/saxe-init.h"
    "/home/daniel/workspace/dynINO/cnn/shadow-params.h"
    "/home/daniel/workspace/dynINO/cnn/simd-functors.h"
    "/home/daniel/workspace/dynINO/cnn/tensor.h"
    "/home/daniel/workspace/dynINO/cnn/timing.h"
    "/home/daniel/workspace/dynINO/cnn/training.h"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/daniel/workspace/dynINO/build/cnn/libcnn.a")
endif()

