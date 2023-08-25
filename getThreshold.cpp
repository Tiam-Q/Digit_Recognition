#include "getThreshold.h"



//��Mat����λ��
int GetMatMidVal(Mat& img)
{
    //�ж�������ǵ�ͨ��ֱ�ӷ���128
    if (img.channels() > 1) return 128;
    int rows = img.rows;
    int cols = img.cols;
    //��������
    float mathists[256] = { 0 };
    //��������0-255�ĸ���
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


//������Ӧ��ֵ����С�����ֵ
void GetMatMinMaxThreshold(Mat& img, int& minval, int& maxval, float sigma)
{
    int midval = GetMatMidVal(img);

    // �������ֵ
    minval = saturate_cast<uchar>((1.0 - sigma) * midval);
    //�������ֵ
    maxval = saturate_cast<uchar>((1.0 + sigma) * midval);
}





