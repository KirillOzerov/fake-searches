#ifndef INFERENCE_H
#define INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <QImage>
#include <vector>

class Inference {
public:
    Inference(const QString& modelPath);
    float analyze(const QImage& image);

private:
    std::vector<float> imageToTensor(const QImage& image, int targetHeight, int targetWidth);
    float runInference(const std::vector<float>& inputTensor);

    Ort::Session session;
    Ort::Env env;
};

#endif // INFERENCE_H
