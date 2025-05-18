#ifndef INFERENCE_H
#define INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <QImage>
#include <vector>
#include <opencv2/opencv.hpp>

struct InferenceResult {
    float clipProbGan;  // Вероятность GAN для CLIP-модели
    float svmProbGan;   // Вероятность GAN для SVM-модели
};

class Inference {
public:
    Inference(const QString& clipModelPath, const QString& svmModelPath, const QString& scalerModelPath);
    InferenceResult analyze(const QImage& image);

private:
    std::vector<float> imageToTensor(const QImage& image, int targetHeight, int targetWidth);
    std::vector<float> extractSaturationFeatures(const cv::Mat& image);
    std::vector<float> normalizeFeatures(const std::vector<float>& features);
    float runClipInference(const std::vector<float>& inputTensor);
    float runSvmInference(const std::vector<float>& inputTensor);

    Ort::Env env;
    Ort::Session clipSession;
    Ort::Session svmSession;
    Ort::Session scalerSession;
    bool useUnderexposed = true; // Для SVM: использовать недоэкспонированные признаки
};

#endif // INFERENCE_H
