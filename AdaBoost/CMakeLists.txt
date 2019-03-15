cmake_minimum_required (VERSION 3.4.1)
project (vjboost)
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
set(OpenCV_DIR D:\\libs\\opencv\\build\\x64\\vc15\\lib)
ENDIF()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
set(Caffe_Dir E:\\Projects\\cpp\\caffe)
set(Caffe_LIB ${Caffe_Dir}/lib/caffe.lib)
ELSE()
find_package(Caffe REQUIRED)
ENDIF()
include_directories(${Caffe_INCLUDE_DIRS})

file(GLOB vjboost_SRC
    "*.h"
    "*.cpp"
)
ADD_EXECUTABLE(vjboost ${vjboost_SRC})
target_link_libraries(vjboost ${OpenCV_LIBS})