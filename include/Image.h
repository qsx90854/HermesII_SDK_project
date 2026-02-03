#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>

//#define _USE_CV_
#ifdef _USE_CV_
#include <opencv2/opencv.hpp>
#endif

class Image {
private:
    std::vector<uint8_t> data;
    
    int w, h, channels;

    // �I�k�B��
    Image erode(int kernelSize) const {
        Image result(w, h, 1);
        int half = kernelSize / 2;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                uint8_t minVal = 255;
                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            minVal = std::min(minVal, at(nx, ny));
                        }
                    }
                }
                result.at(x, y) = minVal;
            }
        }
        return result;
    }

    // ���ȹB��
    Image dilate(int kernelSize) const {
        Image result(w, h, 1);
        int half = kernelSize / 2;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                uint8_t maxVal = 0;
                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            maxVal = std::max(maxVal, at(nx, ny));
                        }
                    }
                }
                result.at(x, y) = maxVal;
            }
        }
        return result;
    }

public:
    Image(int width = 0, int height = 0, int ch = 1) : w(width), h(height), channels(ch) {
        if (width > 0 && height > 0) {
            data.resize(width * height * channels);
        }
    }

    int width() const { return w; }
    int height() const { return h; }
    int getChannels() const { return channels; }
    bool empty() const { return data.empty(); }

    uint8_t& at(int x, int y, int c = 0) {
        return data[(y * w + x) * channels + c];
    }

    const uint8_t& at(int x, int y, int c = 0) const {
        return data[(y * w + x) * channels + c];
    }

    void setData(unsigned char* _data, int dataSize) { data.assign(_data, _data + dataSize); }

    uint8_t* getData() { return data.data(); }
    const uint8_t* getData() const { return data.data(); }

    bool isInside(int x, int y) const {
        return x >= 0 && x < w && y >= 0 && y < h;
    }

    // �ഫ���Ƕ�
    Image toGray() const {
        if (channels == 1) return *this;

        Image gray(w, h, 1);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // BGR to Gray: 0.299*R + 0.587*G + 0.114*B
                int grayVal = static_cast<int>(
                    0.114 * at(x, y, 0) +  // B
                    0.587 * at(x, y, 1) +  // G
                    0.299 * at(x, y, 2)    // R
                    );
                gray.at(x, y) = std::min(255, std::max(0, grayVal));
            }
        }
        return gray;
    }

    // �ഫ��float���� (�Ω�B��)
    Image toFloat() const {
        Image result(w, h, channels);
        // �`�N�G�o�̤��ϥ�uint8_t�s�x�A���b�ϥήɷ��@float�B�z
        result.data = this->data;
        return result;
    }

    // �ƻs���w�ϰ�
    Image getROI(int x_start, int x_end, int y_start, int y_end) const {
        int roi_w = x_end - x_start;
        int roi_h = y_end - y_start;
        Image roi(roi_w, roi_h, channels);

        for (int y = 0; y < roi_h; y++) {
            for (int x = 0; x < roi_w; x++) {
                for (int c = 0; c < channels; c++) {
                    roi.at(x, y, c) = at(x_start + x, y_start + y, c);
                }
            }
        }
        return roi;
    }

    // �ƻs��Ӽv��
    Image clone() const {
        Image result(w, h, channels);
        result.data = this->data;
        return result;
    }

    // �վ�j�p
    Image resize(int new_w, int new_h) const {
        Image result(new_w, new_h, channels);

        float x_ratio = static_cast<float>(w) / new_w;
        float y_ratio = static_cast<float>(h) / new_h;

        for (int y = 0; y < new_h; y++) {
            for (int x = 0; x < new_w; x++) {
                int src_x = static_cast<int>(x * x_ratio);
                int src_y = static_cast<int>(y * y_ratio);

                // �T�O���W�X�d��
                src_x = std::min(src_x, w - 1);
                src_y = std::min(src_y, h - 1);

                for (int c = 0; c < channels; c++) {
                    result.at(x, y, c) = at(src_x, src_y, c);
                }
            }
        }
        return result;
    }

#ifdef _USE_CV_
    // �ഫ��OpenCV Mat (�ȥΩ����)
    cv::Mat toMat() const {
        if (channels == 1) {
            return cv::Mat(h, w, CV_8UC1, const_cast<uint8_t*>(data.data()));
        }
        else if (channels == 3) {
            return cv::Mat(h, w, CV_8UC3, const_cast<uint8_t*>(data.data()));
        }
        return cv::Mat();
    }

    // �qOpenCV Mat�Ы�Image
    static Image fromMat(const cv::Mat& mat) {
        Image img(mat.cols, mat.rows, mat.channels());

        // �ƻs�ƾ�
        for (int y = 0; y < mat.rows; y++) {
            for (int x = 0; x < mat.cols; x++) {
                if (mat.channels() == 3) {
                    cv::Vec3b pixel = mat.at<cv::Vec3b>(y, x);
                    img.at(x, y, 0) = pixel[0]; // B
                    img.at(x, y, 1) = pixel[1]; // G
                    img.at(x, y, 2) = pixel[2]; // R
                }
                else if (mat.channels() == 1) {
                    img.at(x, y) = mat.at<uchar>(y, x);
                }
            }
        }

        return img;
    }
#endif
    void fromUCHAR(unsigned char* _data, int dataSize)
    {
        data.assign(_data, _data + dataSize); 
    }
    // �G�Ȥ�
    Image threshold(int thresh, int maxval = 255) const {
        Image result(w, h, 1);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                result.at(x, y) = (at(x, y) > thresh) ? maxval : 0;
            }
        }
        return result;
    }

    // �κA�Ƕ}�B�� (�I�k�῱��)
    Image morphOpen(int kernelSize = 3) const {
        return erode(kernelSize).dilate(kernelSize);
    }

    // �κA�ǳ��B�� (���ȫ�I�k)
    Image morphClose(int kernelSize = 3) const {
        return dilate(kernelSize).erode(kernelSize);
    }
};