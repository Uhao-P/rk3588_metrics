cmake_minimum_required(VERSION 3.0.0)
project(metrics)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(
    include_libs  
    /usr/include/rga/
    /usr/include/
)

include_directories(${include_libs})



# 添加动态链接库
set(
    dynamic_libs  pthread
    /usr/lib/librknnrt.so
    /usr/lib/aarch64-linux-gnu/librga.so
)


add_executable(classification
    src/classification.cc
)
target_link_libraries(classification
    ${include_libs}
    ${dynamic_libs}
    ${OpenCV_LIBS}
)



add_executable(detections
    src/detections.cc
    src/postprocess.cc
)

target_link_libraries(detections
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)