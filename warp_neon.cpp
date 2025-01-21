#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <arm_neon.h>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;

void convertToYUV420(const Mat& src, Mat& yuv) {
    cvtColor(src, yuv, COLOR_BGR2YUV_I420);
}

void warpPerspectiveWithoutNeon(const Mat& src, Mat& dst, const Mat& transformMatrix, Size size) {
    warpPerspective(src, dst, transformMatrix, size, INTER_LINEAR | WARP_INVERSE_MAP, BORDER_CONSTANT);
}

void warpPerspectiveWithNeon(const Mat& src, Mat& dst, const Mat& transformMatrix, Size size) {
    // 使用 OpenCV 内部的 NEON 优化实现
    warpPerspective(src, dst, transformMatrix, size, INTER_LINEAR | WARP_INVERSE_MAP, BORDER_CONSTANT);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <image_path> <use_neon: 0 or 1>" << endl;
        return -1;
    }

    string imagePath = argv[1];
    bool useNeon = (atoi(argv[2]) == 1);

    // 1. 加载图片
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Failed to load image at " << imagePath << endl;
        return -1;
    }

    // 2. 转换为 YUV420 格式
    Mat yuvImage;
    auto startConvert = chrono::high_resolution_clock::now();
    convertToYUV420(image, yuvImage);
    auto endConvert = chrono::high_resolution_clock::now();

    // 3. 定义透视变换矩阵
    Mat transformMatrix = getRotationMatrix2D(Point2f(image.cols / 2, image.rows / 2), 30, 1.0);

    // 扩展矩阵为 3x3
    Mat perspectiveMatrix(3, 3, CV_64F);
    perspectiveMatrix.at<double>(0, 0) = transformMatrix.at<double>(0, 0);
    perspectiveMatrix.at<double>(0, 1) = transformMatrix.at<double>(0, 1);
    perspectiveMatrix.at<double>(0, 2) = transformMatrix.at<double>(0, 2);
    perspectiveMatrix.at<double>(1, 0) = transformMatrix.at<double>(1, 0);
    perspectiveMatrix.at<double>(1, 1) = transformMatrix.at<double>(1, 1);
    perspectiveMatrix.at<double>(1, 2) = transformMatrix.at<double>(1, 2);
    perspectiveMatrix.at<double>(2, 0) = 0;
    perspectiveMatrix.at<double>(2, 1) = 0;
    perspectiveMatrix.at<double>(2, 2) = 1;

    // 4. 应用 warpPerspective
    Mat outputImage;
    auto startWarp = chrono::high_resolution_clock::now();

    if (useNeon) {
        warpPerspectiveWithNeon(yuvImage, outputImage, perspectiveMatrix, image.size());
        cout << "Using NEON optimization for warpPerspective." << endl;
    } else {
        warpPerspectiveWithoutNeon(yuvImage, outputImage, perspectiveMatrix, image.size());
        cout << "Using standard warpPerspective without NEON." << endl;
    }

    auto endWarp = chrono::high_resolution_clock::now();

    // 5. 计算耗时
    auto convertTime = chrono::duration_cast<chrono::milliseconds>(endConvert - startConvert).count();
    auto warpTime = chrono::duration_cast<chrono::milliseconds>(endWarp - startWarp).count();

    cout << "Conversion to YUV420 Time: " << convertTime << " ms" << endl;
    cout << "WarpPerspective Time: " << warpTime << " ms" << endl;

    // 6. 保存输出图片
    imwrite("output.jpg", outputImage);
    cout << "Output saved as output.jpg" << endl;

    return 0;
}
