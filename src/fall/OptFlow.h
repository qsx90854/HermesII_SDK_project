#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <iostream>
#include <chrono>
#include "Image.h"

// 基本資料結構
struct Point2f {
    float x, y;
    Point2f(float x = 0, float y = 0) : x(x), y(y) {}
};

struct Track {
    std::vector<Point2f> points;
    int id;
};

struct Object {
    int id;
    Point2f centroid;
    Point2f movement;
    float area;
    std::vector<Point2f> points;
    std::vector<Track> tracks;
};

// 圖像金字塔
class ImagePyramid {
private:
    std::vector<Image> levels;

public:
    ImagePyramid() {}
    ImagePyramid(const Image& img, int maxLevel) {
        levels.push_back(img);

        for (int level = 1; level <= maxLevel; ++level) {
            int newWidth = levels[level - 1].width() / 2;
            int newHeight = levels[level - 1].height() / 2;

            if (newWidth < 2 || newHeight < 2) break;

            Image downsampled(newWidth, newHeight);
            downsample(levels[level - 1], downsampled);
            levels.push_back(downsampled);
        }
    }

    const Image& getLevel(int level) const {
        return levels[std::min(level, (int)levels.size() - 1)];
    }

    int numLevels() const { return levels.size(); }

private:
    void downsample(const Image& src, Image& dst) {
        for (int y = 0; y < dst.height(); ++y) {
            for (int x = 0; x < dst.width(); ++x) {
                int sx = x * 2;
                int sy = y * 2;
                float sum = 0;
                int count = 0;

                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        if (src.isInside(sx + dx, sy + dy)) {
                            sum += src.at(sx + dx, sy + dy);
                            count++;
                        }
                    }
                }
                dst.at(x, y) = count > 0 ? sum / count : 0;
            }
        }
    }
};

// 計算圖像梯度
class ImageGradient {
public:
    Image Ix, Iy;

    ImageGradient() {}
    ImageGradient(const Image& img) : Ix(img.width(), img.height()), Iy(img.width(), img.height()) {
        if (img.getChannels() > 1) {
            Image img_gray = img.toGray();
            computeGradients(img_gray);
        }
        else {
            computeGradients(img);
        }
#ifdef _USE_CV__
        cv::Mat cv_img = img.toMat();
        cv::Mat cv_Ix = Ix.toMat();
        cv::Mat cv_Iy = Iy.toMat();
        std::cout << "ImageGradient finished" << std::endl;
#endif
    }

private:
    void computeGradients(const Image& img) {
        // Sobel 梯度
        for (int y = 1; y < img.height() - 1; ++y) {
            for (int x = 1; x < img.width() - 1; ++x) {
                float gx = -img.at(x - 1, y - 1) + img.at(x + 1, y - 1) +
                    -2 * img.at(x - 1, y) + 2 * img.at(x + 1, y) +
                    -img.at(x - 1, y + 1) + img.at(x + 1, y + 1);

                float gy = -img.at(x - 1, y - 1) - 2 * img.at(x, y - 1) - img.at(x + 1, y - 1) +
                    img.at(x - 1, y + 1) + 2 * img.at(x, y + 1) + img.at(x + 1, y + 1);

                Ix.at(x, y) = gx / 8.0f;
                Iy.at(x, y) = gy / 8.0f;
            }
        }
    }
};