#include "inference.h"
#include <algorithm>
#include <cmath>
#include <QDebug>
#include <QFile>

Inference::Inference(const QString& clipModelPath, const QString& svmModelPath, const QString& scalerModelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "Inference"), clipSession(nullptr), svmSession(nullptr), scalerSession(nullptr) {
    qDebug() << "Starting Inference constructor...";
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);

    // Загрузка CLIP-модели
    QFile clipFile(clipModelPath);
    if (!clipFile.exists()) {
        qDebug() << "CLIP model file does not exist:" << clipModelPath;
        throw std::runtime_error("CLIP model file not found");
    }
    if (!clipFile.open(QIODevice::ReadOnly)) {
        qDebug() << "Failed to open CLIP model:" << clipModelPath << "Error:" << clipFile.errorString();
        throw std::runtime_error("Failed to open CLIP model");
    }
    QByteArray clipModelData = clipFile.readAll();
    qDebug() << "CLIP model data read, size:" << clipModelData.size() << "bytes";
    clipSession = Ort::Session(env, clipModelData.constData(), clipModelData.size(), sessionOptions);
    qDebug() << "CLIP model loaded:" << clipModelPath;

    // Загрузка SVM-модели
    QFile svmFile(svmModelPath);
    if (!svmFile.exists()) {
        qDebug() << "SVM model file does not exist:" << svmModelPath;
        throw std::runtime_error("SVM model file not found");
    }
    if (!svmFile.open(QIODevice::ReadOnly)) {
        qDebug() << "Failed to open SVM model:" << svmModelPath << "Error:" << svmFile.errorString();
        throw std::runtime_error("Failed to open SVM model");
    }
    QByteArray svmModelData = svmFile.readAll();
    qDebug() << "SVM model data read, size:" << svmModelData.size() << "bytes";
    svmSession = Ort::Session(env, svmModelData.constData(), svmModelData.size(), sessionOptions);
    qDebug() << "SVM model loaded:" << svmModelPath;

    // Загрузка нормализатора для SVM
    QFile scalerFile(scalerModelPath);
    if (!scalerFile.exists()) {
        qDebug() << "Scaler model file does not exist:" << scalerModelPath;
        throw std::runtime_error("Scaler model file not found");
    }
    if (!scalerFile.open(QIODevice::ReadOnly)) {
        qDebug() << "Failed to open scaler model:" << scalerModelPath << "Error:" << scalerFile.errorString();
        throw std::runtime_error("Failed to open scaler model");
    }
    QByteArray scalerModelData = scalerFile.readAll();
    qDebug() << "Scaler model data read, size:" << scalerModelData.size() << "bytes";
    scalerSession = Ort::Session(env, scalerModelData.constData(), scalerModelData.size(), sessionOptions);
    qDebug() << "Scaler model loaded:" << scalerModelPath;
}

std::vector<float> Inference::imageToTensor(const QImage& image, int targetHeight, int targetWidth) {
    // Масштабируем изображение, сохраняя пропорции, чтобы минимальная сторона была 224
    int originalWidth = image.width();
    int originalHeight = image.height();
    int newWidth, newHeight;

    if (originalWidth > originalHeight) {
        newHeight = targetHeight; // 224
        newWidth = (targetHeight * originalWidth) / originalHeight;
    } else {
        newWidth = targetWidth; // 224
        newHeight = (targetWidth * originalHeight) / originalWidth;
    }

    QImage scaledImage = image.scaled(newWidth, newHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // Обрезаем центральную часть 224x224
    int xOffset = (scaledImage.width() - targetWidth) / 2;
    int yOffset = (scaledImage.height() - targetHeight) / 2;
    QImage croppedImage = scaledImage.copy(xOffset, yOffset, targetWidth, targetHeight);

    QImage rgbImage = croppedImage.convertToFormat(QImage::Format_RGB888);

    // Инициализация тензора в формате CHW
    std::vector<float> tensor(3 * targetHeight * targetWidth);

    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    // Заполняем тензор в формате CHW (Channels, Height, Width)
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < targetHeight; ++y) {
            for (int x = 0; x < targetWidth; ++x) {
                QRgb pixel = rgbImage.pixel(x, y);
                float value;
                if (c == 0) value = static_cast<float>(qRed(pixel)) / 255.0f;   // R
                else if (c == 1) value = static_cast<float>(qGreen(pixel)) / 255.0f; // G
                else value = static_cast<float>(qBlue(pixel)) / 255.0f;      // B

                // Нормализация
                value = (value - mean[c]) / std[c];

                // Заполняем тензор
                tensor[c * targetHeight * targetWidth + y * targetWidth + x] = value;
            }
        }
    }

    return tensor;
}

std::vector<float> Inference::extractSaturationFeatures(const cv::Mat& image) {
    if (image.empty()) {
        qDebug() << "Empty image provided to extractSaturationFeatures";
        return {};
    }

    // Преобразование в градации серого
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 1) {
        grayImage = image;
    } else {
        qDebug() << "Unsupported image format";
        return {};
    }

    int totalPixels = grayImage.rows * grayImage.cols;
    std::vector<int> overexposedThresholds = {240, 245, 250, 255};
    std::vector<float> features;

    // Переэкспонированные признаки
    for (int thresh : overexposedThresholds) {
        int count = cv::countNonZero(grayImage >= thresh);
        features.push_back(static_cast<float>(count) / totalPixels);
    }

    // Недоэкспонированные признаки
    if (useUnderexposed) {
        std::vector<int> underexposedThresholds = {0, 5, 10, 15};
        for (int thresh : underexposedThresholds) {
            int count = cv::countNonZero(grayImage <= thresh);
            features.push_back(static_cast<float>(count) / totalPixels);
        }
    }

    qDebug() << "Extracted features:" << features.size() << "values";
    return features;
}

std::vector<float> Inference::normalizeFeatures(const std::vector<float>& features) {
    if (features.empty()) {
        qDebug() << "Empty features provided to normalizeFeatures";
        return {};
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Подготовка входного тензора
    std::vector<int64_t> inputDims = {1, static_cast<int64_t>(features.size())};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(features.data()), features.size(), inputDims.data(), inputDims.size()
        );

    // Получение имени входа и выхода
    auto inputName = scalerSession.GetInputNameAllocated(0, allocator);
    const char* inputNamePtr = inputName.get();
    const char* inputNames[] = {inputNamePtr};

    auto outputName = scalerSession.GetOutputNameAllocated(0, allocator);
    const char* outputNamePtr = outputName.get();
    const char* outputNames[] = {outputNamePtr};

    // Выполнение нормализации
    std::vector<float> scaledFeatures(features.size());
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, scaledFeatures.data(), scaledFeatures.size(), inputDims.data(), inputDims.size()
        );

    scalerSession.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, &outputTensor, 1);

    qDebug() << "Normalized features:" << scaledFeatures.size() << "values";
    return scaledFeatures;
}

float Inference::runClipInference(const std::vector<float>& inputTensor) {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Выведите входной тензор
    qDebug() << "First 5 input values:";
    for (int i = 0; i < 5; ++i) {
        qDebug() << inputTensor[i];
    }

    auto inputTypeInfo = clipSession.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims[0] == -1) inputDims[0] = 1;

    Ort::Value inputTensorObj = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputTensor.data()), inputTensor.size(), inputDims.data(), inputDims.size()
        );

    auto outputTypeInfo = clipSession.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims[0] == -1) outputDims[0] = 1;

    size_t outputSize = 1;
    for (int64_t dim : outputDims) outputSize *= dim;
    std::vector<float> outputTensorValues(outputSize);
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputSize, outputDims.data(), outputDims.size()
        );

    auto inputName = clipSession.GetInputNameAllocated(0, allocator);
    auto outputName = clipSession.GetOutputNameAllocated(0, allocator);
    const char* inputNames[] = {inputName.get()};
    const char* outputNames[] = {outputName.get()};

    clipSession.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensorObj, 1, outputNames, &outputTensor, 1);

    float logit = outputTensorValues[0];
    float prob = 1.0f / (1.0f + std::exp(-logit));
    qDebug() << "Logit:" << logit;
    qDebug() << "Probability:" << prob;
    return prob;
}

float Inference::runSvmInference(const std::vector<float>& scaledFeatures) {
    std::cout << "SVM input size: " << scaledFeatures.size() << std::endl;
    std::cout << "SVM input values: ";
    for (const auto& val : scaledFeatures) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    if (!svmSession) {
        std::cerr << "SVM session is not initialized!" << std::endl;
        throw std::runtime_error("SVM session is not initialized");
    }

    try {
        std::vector<int64_t> inputDims = {1, static_cast<int64_t>(scaledFeatures.size())};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(scaledFeatures.data()), scaledFeatures.size(), inputDims.data(), inputDims.size());

        const char* inputNames[] = {"float_input"}; // Исправлено с "input" на "float_input"
        const char* outputNames[] = {"probabilities"}; // Используем "probabilities" для вероятностей

        std::cout << "Executing SVM inference..." << std::endl;
        auto outputTensors = svmSession.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        // Для SVM с двумя классами probabilities возвращает [prob_class_0, prob_class_1]
        // Берем вероятность для класса 1 (например, "GAN")
        float svmProbGan = outputData[1]; // Индекс 1 для класса 1
        std::cout << "SVM output (probability for class 1): " << svmProbGan << std::endl;

        return svmProbGan;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error in SVM inference: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in SVM inference: " << e.what() << std::endl;
        throw;
    }
}

InferenceResult Inference::analyze(const QImage& image) {
    InferenceResult result = { -1.0f, -1.0f };

    // CLIP: обработка изображения
    std::vector<float> clipTensor = imageToTensor(image, 224, 224);
    if (!clipTensor.empty()) {
        result.clipProbGan = runClipInference(clipTensor);
    }

    // SVM: конвертация QImage в cv::Mat
    cv::Mat cvImage;
    if (image.format() == QImage::Format_RGB32 || image.format() == QImage::Format_ARGB32) {
        cvImage = cv::Mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), image.bytesPerLine());
        cv::cvtColor(cvImage, cvImage, cv::COLOR_BGRA2BGR);
    } else if (image.format() == QImage::Format_RGB888) {
        cvImage = cv::Mat(image.height(), image.width(), CV_8UC3, const_cast<uchar*>(image.bits()), image.bytesPerLine());
        cv::cvtColor(cvImage, cvImage, cv::COLOR_RGB2BGR);
    } else {
        qDebug() << "Unsupported QImage format for SVM";
        return result;
    }

    // SVM: извлечение и нормализация признаков
    std::vector<float> features = extractSaturationFeatures(cvImage);
    if (!features.empty()) {
        std::vector<float> scaledFeatures = normalizeFeatures(features);
        if (!scaledFeatures.empty()) {
            std::cout << "Running SVM inference..." << std::endl;
            result.svmProbGan = runSvmInference(scaledFeatures);
            std::cout << "SVM inference completed." << std::endl;
        }
    }

    return result;
}
