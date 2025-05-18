#include "mainwindow.h"
#include <QVBoxLayout>
#include <QFileDialog>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    qDebug() << "Starting MainWindow constructor...";
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    imageLabel = new QLabel(this);
    imageLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(imageLabel);

    loadButton = new QPushButton("Загрузить изображение", this);
    analyzeButton = new QPushButton("Анализировать", this);
    clearButton = new QPushButton("Очистить", this);
    layout->addWidget(loadButton);
    layout->addWidget(analyzeButton);
    layout->addWidget(clearButton);

    resultText = new QTextEdit(this);
    resultText->setReadOnly(true);
    layout->addWidget(resultText);

    inference = new Inference(
        "C:/Users/kiril/Documents/test_onnx/models/clipVit_model_opset15.onnx",
        "C:/Users/kiril/Documents/test_onnx/models/svm_model_full.onnx",
        "C:/Users/kiril/Documents/test_onnx/models/scaler_full.onnx"
        );

    connect(loadButton, &QPushButton::clicked, this, &MainWindow::loadImage);
    connect(analyzeButton, &QPushButton::clicked, this, &MainWindow::analyzeImage);
    connect(clearButton, &QPushButton::clicked, this, &MainWindow::clear);

    setWindowTitle("Image Analysis");
    resize(300, 400);
    qDebug() << "MainWindow constructor finished.";
}

MainWindow::~MainWindow() {
    delete inference;
}

void MainWindow::loadImage() {
    currentImagePath = QFileDialog::getOpenFileName(this, "Выберите изображение", "", "Images (*.png *.jpg)");
    if (!currentImagePath.isEmpty()) {
        currentImage = QImage(currentImagePath);
        imageLabel->setPixmap(QPixmap::fromImage(currentImage).scaled(200, 200, Qt::KeepAspectRatio));
        resultText->clear();
    }
}

void MainWindow::analyzeImage() {
    if (currentImage.isNull()) {
        resultText->setText("Сначала загрузите изображение!");
        return;
    }

    InferenceResult result = inference->analyze(currentImage);
    if (result.clipProbGan >= 0 && result.svmProbGan >= 0) {
        QString output;
        output += QString("Модель 1: %1 (Скорее всего %2)\n")
                      .arg(result.clipProbGan, 0, 'f', 4)
                      .arg(result.clipProbGan > 0.5 ? "сгенерировано" : "настоящее");
        output += QString("Модель 2: %1 (Скорее всего %2)")
                      .arg(result.svmProbGan, 0, 'f', 4)
                      .arg(result.svmProbGan > 0.5 ? "сгенерировано" : "настоящее");
        resultText->setText(output);
    } else {
        resultText->setText("Ошибка при анализе.");
    }
}

void MainWindow::clear() {
    currentImage = QImage();
    imageLabel->clear();
    resultText->clear();
}
