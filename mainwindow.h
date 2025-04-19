#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include "inference.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void loadImage();
    void analyzeImage();
    void clear();

private:
    QLabel *imageLabel;
    QPushButton *loadButton;
    QPushButton *analyzeButton;
    QPushButton *clearButton;
    QTextEdit *resultText;
    Inference *inference;
    QImage currentImage;
    QString currentImagePath;
};

#endif // MAINWINDOW_H
