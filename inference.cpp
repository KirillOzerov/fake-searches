#include "inference.h"
#include <algorithm>
#include <cmath>
#include <QDebug>
#include <QFile>

Inference::Inference(const QString& modelPath) : env(ORT_LOGGING_LEVEL_WARNING, "test"), session(nullptr) {
    qDebug() << "Starting Inference constructor...";
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);

    QFile file(modelPath);
    if (!file.exists()) {
        qDebug() << "File does not exist:" << modelPath;
        return;
    }
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "Failed to open model:" << modelPath << "Error:" << file.errorString();
        return;
    }
    QByteArray modelData = file.readAll();
    qDebug() << "Model data read, size:" << modelData.size() << "bytes";
    session = Ort::Session(env, modelData.constData(), modelData.size(), sessionOptions);
    qDebug() << "Model loaded:" << modelPath;
}

std::vector<float> Inference::imageToTensor(const QImage& image, int targetHeight, int targetWidth) {
    int originalWidth = image.width();
    int originalHeight = image.height();
    int cropSize = std::min(originalWidth, originalHeight);
    int xOffset = (originalWidth - cropSize) / 2;
    int yOffset = (originalHeight - cropSize) / 2;
    QImage croppedImage = image.copy(xOffset, yOffset, cropSize, cropSize);

    QImage resizedImage = croppedImage.scaled(targetWidth, targetHeight, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QImage rgbImage = resizedImage.convertToFormat(QImage::Format_RGB888);

    std::vector<float> tensor;
    tensor.reserve(3 * targetHeight * targetWidth);

    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < targetHeight; ++y) {
        for (int x = 0; x < targetWidth; ++x) {
            QRgb pixel = rgbImage.pixel(x, y);
            float r = static_cast<float>(qRed(pixel)) / 255.0f;
            float g = static_cast<float>(qGreen(pixel)) / 255.0f;
            float b = static_cast<float>(qBlue(pixel)) / 255.0f;

            r = (r - mean[0]) / std[0];
            g = (g - mean[1]) / std[1];
            b = (b - mean[2]) / std[2];

            tensor.push_back(r);
            tensor.push_back(g);
            tensor.push_back(b);
        }
    }

    return tensor;
}

float Inference::runInference(const std::vector<float>& inputTensor) {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Выведите входной тензор
    qDebug() << "First 5 input values:";
    for (int i = 0; i < 5; ++i) {
        qDebug() << inputTensor[i];
    }

    auto inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims[0] == -1) inputDims[0] = 1;

    Ort::Value inputTensorObj = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputTensor.data()), inputTensor.size(), inputDims.data(), inputDims.size()
        );

    auto outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims[0] == -1) outputDims[0] = 1;

    size_t outputSize = 1;
    for (int64_t dim : outputDims) outputSize *= dim;
    std::vector<float> outputTensorValues(outputSize);
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputSize, outputDims.data(), outputDims.size()
        );

    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);
    const char* inputNames[] = {inputName.get()};
    const char* outputNames[] = {outputName.get()};

    session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensorObj, 1, outputNames, &outputTensor, 1);

    float logit = outputTensorValues[0];
    float prob = 1.0f / (1.0f + std::exp(-logit));
    qDebug() << "Logit:" << logit;
    qDebug() << "Probability:" << prob;
    return prob;
}

float Inference::analyze(const QImage& image) {
    std::vector<float> inputTensor = imageToTensor(image, 224, 224);
    return runInference(inputTensor);
}
