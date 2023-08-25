#include "getThreshold.h"



//求Mat的中位数
int GetMatMidVal(Mat& img)
{
    //判断如果不是单通道直接返回128
    if (img.channels() > 1) return 128;
    int rows = img.rows;
    int cols = img.cols;
    //定义数组
    float mathists[256] = { 0 };
    //遍历计算0-255的个数
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int val = img.at<uchar>(row, col);
            mathists[val]++;
        }
    }


    int calcval = rows * cols / 2;
    int tmpsum = 0;
    for (int i = 0; i < 255; ++i) {
        tmpsum += mathists[i];
        if (tmpsum > calcval) {
            return i;
        }
    }
    return 0;
}


//求自适应阈值的最小和最大值
void GetMatMinMaxThreshold(Mat& img, int& minval, int& maxval, float sigma)
{
    int midval = GetMatMidVal(img);

    // 计算低阈值
    minval = saturate_cast<uchar>((1.0 - sigma) * midval);
    //计算高阈值
    maxval = saturate_cast<uchar>((1.0 + sigma) * midval);
}





