cmake_minimum_required(VERSION 3.6)
# It's naughty to put these here as CMAKE is cross platform - they now live in the IDE settings
# They can be set in the CLION settings
#set(CMAKE_CXX_COMPILER /usr/bin/g++)
#set(CMAKE_C_COMPILER /usr/bin/gcc)
#
#set(CMAKE_CXX_FLAGS “${CMAKE_CXX_FLAGS} -std=c++11 -undefined dynamic_lookup“)

project(tf_customer_opps)
include_directories(BEFORE /Users/ben.chamberlain/anaconda/envs/tensorflow/lib/python2.7/site-packages/tensorflow/include)

set(CMAKE_CXX_STANDARD 11)


#set(SOURCE_FILES zero_out.cc word2vec_ops.cpp generate_batch.cpp test_harness.cpp test_harness.h)
set(SOURCE_FILES test_harness.cpp)
add_executable(tf_customer_opps ${SOURCE_FILES})