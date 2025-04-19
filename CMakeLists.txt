cmake_minimum_required(VERSION 3.14)
project(test_onnx LANGUAGES CXX)

# Настройка Qt
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# Включаем автоматическую обработку .ui файлов
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Добавляем исходные файлы
set(SOURCES
    main.cpp
    mainwindow.cpp
)

set(HEADERS
    mainwindow.h
)

set(FORMS
    mainwindow.ui
)

# Создаем исполняемый файл
add_executable(test_onnx ${SOURCES} ${HEADERS} ${FORMS}
    mainwindow.ui
    models/my_super_puper_model.onnx
    models/clipVit_model_opset15.onnx
    aadlygmazf.png
    aadlygmazf.jpg
    aaksjqrtil.jpg

    inference.h inference.cpp
    me_smol.jpg

)

# Подключаем модули Qt
target_link_libraries(test_onnx PRIVATE Qt6::Core Qt6::Widgets)

# Настройка ONNX Runtime
set(ONNX_RUNTIME_DIR "C:/diploma/onnxruntime-win-x64-1.20.1")

# Добавляем путь к заголовочным файлам ONNX Runtime
target_include_directories(test_onnx PRIVATE "${ONNX_RUNTIME_DIR}/include")

# Добавляем путь к библиотеке ONNX Runtime
target_link_directories(test_onnx PRIVATE "${ONNX_RUNTIME_DIR}/lib")

# Подключаем библиотеку ONNX Runtime
target_link_libraries(test_onnx PRIVATE onnxruntime)

# Копируем onnxruntime.dll в папку с исполняемым файлом
add_custom_command(TARGET test_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${ONNX_RUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:test_onnx>
)
