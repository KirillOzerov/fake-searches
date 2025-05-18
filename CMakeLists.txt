cmake_minimum_required(VERSION 3.14)
project(test_onnx LANGUAGES CXX)

# Настройка Qt
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)
if(Qt6_FOUND)
    message(STATUS "Qt6 found: ${Qt6_DIR}")
else()
    message(FATAL_ERROR "Qt6 not found")
endif()

# Включаем автоматическую обработку moc
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Настройка OpenCV
set(OpenCV_DIR "C:/Qt6.8/opencv/build")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_DIR}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found in ${OpenCV_DIR}")
endif()

# Добавляем пути к заголовкам OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Добавляем исходные файлы
set(SOURCES
    main.cpp
    mainwindow.cpp
    inference.cpp
)

set(HEADERS
    mainwindow.h
    inference.h
)

# Создаем исполняемый файл
add_executable(test_onnx ${SOURCES} ${HEADERS}
    models/clipVit_model_opset15.onnx
    models/svm_model_full.onnx
    models/scaler_full.onnx
    aadlygmazf.png
    aadlygmazf.jpg
    aaksjqrtil.jpg
    me_smol.jpg
)

# Настройка ONNX Runtime
set(ONNX_RUNTIME_DIR "C:/diploma/onnxruntime-win-x64-1.20.1")
if(EXISTS "${ONNX_RUNTIME_DIR}")
    message(STATUS "ONNX Runtime found: ${ONNX_RUNTIME_DIR}")
else()
    message(FATAL_ERROR "ONNX Runtime not found in ${ONNX_RUNTIME_DIR}")
endif()

# Добавляем путь к заголовочным файлам ONNX Runtime
target_include_directories(test_onnx PRIVATE "${ONNX_RUNTIME_DIR}/include")

# Добавляем путь к библиотеке ONNX Runtime
target_link_directories(test_onnx PRIVATE "${ONNX_RUNTIME_DIR}/lib")

# Подключаем библиотеки Qt, ONNX Runtime, OpenCV
target_link_libraries(test_onnx PRIVATE Qt6::Core Qt6::Widgets onnxruntime ${OpenCV_LIBS})

# Копируем onnxruntime.dll в папку с исполняемым файлом
add_custom_command(TARGET test_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${ONNX_RUNTIME_DIR}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:test_onnx>"
    COMMENT "Copying ONNX Runtime DLL to output directory"
)

# Копируем OpenCV DLL в папку с исполняемым файлом (для Debug)
add_custom_command(TARGET test_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "C:/Qt6.8/opencv/build/x64/vc16/bin/opencv_world4110d.dll"
        "$<TARGET_FILE_DIR:test_onnx>"
    COMMENT "Copying OpenCV DLL to output directory"
)
