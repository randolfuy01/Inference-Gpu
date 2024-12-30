cmake_minimum_required(VERSION 3.10)
project(InferenceBackend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH /opt/homebrew/Cellar/pytorch/2.5.1) 

find_package(Torch REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/compute
    ${CMAKE_SOURCE_DIR}/interface
    ${CMAKE_SOURCE_DIR}/model
)

add_executable(InferenceBackend
    model/ModelLoader.cpp
    model/PreProcessor.cpp
    interface/SomeInterfaceFile.cpp
    main.cpp
)

target_link_libraries(InferenceBackend "${TORCH_LIBRARIES}")
set_property(TARGET InferenceBackend PROPERTY CXX_STANDARD 17)