# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(face_tracker)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# Find and add Pthread package
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Enable the debug mode
# set(CMAKE_BUILD_TYPE Debug)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(face_tracker face_tracker.cpp)

# Link your application with OpenCV libraries
target_link_libraries(face_tracker PRIVATE ${OpenCV_LIBS})

# Link your application with pthread libraries
target_link_libraries( face_tracker PRIVATE pthread )