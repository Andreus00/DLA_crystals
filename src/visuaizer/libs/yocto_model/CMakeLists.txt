#project(cmake_and_cuda LANGUAGES CXX CUDA)


add_library(yocto_model 
  yocto_model.h 
  yocto_model.cpp
  ext/perlin-noise/noise1234.cpp)
  #../cuda/parallel_dla_cuda.cu)
######################
#set_property(TARGET yocto_model PROPERTY CUDA_ARCHITECTURES ON)

# set_property(TARGET ../cuda/parallel_dla_cuda.cu
#              PROPERTY CUDA_SEPARABLE_COMPILATION ON)


######################
target_include_directories(yocto_model PRIVATE .)
target_include_directories(yocto_model PRIVATE ./ext)
target_include_directories(yocto_model PUBLIC ${CMAKE_SOURCE_DIR}/libs)
set_target_properties(yocto_model PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

target_link_libraries(yocto_model PUBLIC yocto)
#target_compile_options(yocto_model PUBLIC -fopenmp)
# warning flags


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(yocto_model PUBLIC OpenMP::OpenMP_CXX)
endif()

if(APPLE)
  target_compile_options(yocto_model PUBLIC -Wall -Wconversion -Wno-sign-conversion -Wno-implicit-float-conversion)
endif(APPLE)
if(MSVC)
target_compile_options(yocto_model PUBLIC /D_CRT_SECURE_NO_WARNINGS /wd4018 /wd4244 /wd4305 /wd4800 /wd4267)
target_compile_options(yocto_model PUBLIC /EHsc)
endif(MSVC)
