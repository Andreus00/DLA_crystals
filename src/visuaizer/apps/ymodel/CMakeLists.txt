add_executable(ymodel  ymodel.cpp)

set_target_properties(ymodel  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories(ymodel  PRIVATE ${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(ymodel  yocto yocto_model)

