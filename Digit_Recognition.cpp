///*
//                   _ooOoo_
//                  o8888888o
//                  88" . "88
//                  (| -_- |)
//                  O\  =  /O
//               ____/`---'\____
//             .'  \\|     |//  `.
//            /  \\|||  :  |||//  \
//           /  _||||| -:- |||||-  \
//           |   | \\\  -  /// |   |
//           | \_|  ''\---/''  |   |
//           \  .-\__  `-`  ___/-. /
//         ___`. .'  /--.--\  `. . __
//      ."" '<  `.___\_<|>_/___.'  >'"".
//     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//     \  \ `-.   \_ __\ /__ _/   .-` /  /
//======`-.____`-.___\_____/___.-`____.-'======
//                   `=---='
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//         佛祖保佑       永无BUG
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//作者：2131277807@qq.com
//时间：2023\08\25
//编译器：visual studio
//环境：opencv3.4.4
//描述：根据HSV颜色空间对车牌进行定位，之后利用训练好的SVM模型对车牌进行精准筛选，
//      再利用直线霍夫变换，检测倾斜角度，进行倾斜校正，之后使用大津法对不同底色的
//      车牌进行不同方法的二值化，得到统一的白字黑底。
// SVM模型训练在：https://github.com/Tiam-Q/classify.git
// 详情：https://blog.csdn.net/ltj5201314/article/details/132498010
//*/

#include <opencv2/opencv.hpp>  
#include<iostream>  
#include<stdio.h>
#include <string>
#include <fstream>
#include "CLbp.h"
#include "getThreshold.h"
#include<math.h>


using namespace std;
using namespace cv;
using namespace cv::ml;


#define     _CRT_SECURE_NO_WARNINGS


//大津法计算二值化阈值
double Otsu(Mat& image);
//判断车牌颜色 0:蓝色 1：黄色 2：绿色
int colorJudge(Mat img);



// 训练集数组，将训练集对应于一个数组
string Test_Arr[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
                      "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", \
                      "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", \
                      "W", "X", "Y", "Z", \
                      "川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", \
                      "冀", "津", "京", "吉", "辽", "鲁", "蒙", "闽", \
                      "宁", "青", "琼", "陕", "苏", "晋", "皖", "湘", \
                      "新", "豫", "渝", "粤", "云", "藏", "浙" \
                   };


#define     imgRows      36      //图像行数
#define     imgCols      136     //图像列数


int main()
{
    //加载SVM模型
    string modelPath = "C:/Users/Tiam/Desktop/Digit_Recognition2/svm.xml";
    Ptr<ml::SVM>svmpre = ml::SVM::load(modelPath);
    //加载ann模型
    modelPath = "C:/Users/Tiam/Desktop/Digit_Recognition2/mnist_ann.xml";
    cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::StatModel::load<cv::ml::ANN_MLP>(modelPath);

    Mat grayImg;
    Mat blurImg;
    Mat canImg;
    Mat binaryImg;
    Mat closeImg;
    Mat openImg;
    Mat stretchGrayImg;
    Mat absdiffImg;
    Mat hsvImg;


    //训练集路径
    string carPath = "C:\\Users\\Tiam\\Desktop\\Digit_Recognition2\\image\\car_picture";
    vector<cv::String> imagePathList;
    //读取路径下所有图片
    glob(carPath, imagePathList);

    hsvImg = Mat::zeros(Size(700, 450), CV_8UC1);

    for (int i = 0; i < imagePathList.size(); i++)
    {
        //读取
        auto img = imread(imagePathList[i]);
        //统一图片大小
        resize(img, img, Size(700, 450), 0, 0, INTER_CUBIC/*CV_INTER_AREA*/);
        //高斯模糊
        GaussianBlur(img, blurImg, Size(3, 3), -13);
        //灰度化
        cvtColor(blurImg, grayImg, COLOR_BGR2GRAY);
        //灰度拉伸
        normalize(grayImg, stretchGrayImg, 0, 255, cv::NORM_MINMAX);

        //Y方向开运算
        Mat element = getStructuringElement(MORPH_RECT, Size(1, 13));
        morphologyEx(stretchGrayImg, openImg, MORPH_OPEN, element);
        //X方向
        element = getStructuringElement(MORPH_RECT, Size(13, 1));
        morphologyEx(openImg, openImg, MORPH_OPEN, element);
        //图像差分
        absdiff(stretchGrayImg, openImg, absdiffImg);

        //转换HSV空间
        cvtColor(img, hsvImg, COLOR_BGR2HSV);
        //颜色过滤
        for (int x = 0; x < hsvImg.rows; x++)
        {
            for (int y = 0; y < hsvImg.cols; y++)
            {
                int H = hsvImg.at<Vec3b>(x, y)[0];
                int S = hsvImg.at<Vec3b>(x, y)[1];
                int V = hsvImg.at<Vec3b>(x, y)[2];
                //蓝色
                if ((H >= 100) && (H <= 120) && (S >= 43) && (V >= 46))
                {


                }
                
                //黄色
                else if ((H >= 19) && (H <= 26) && (S >= 43) && (V >= 46))
                {


                }
                //绿色
                else if ((H >= 35) && (H <= 77) && (S >= 43) && (V >= 46))
                {


                }
                else
                {
                    absdiffImg.at<uchar>(x, y) = 0;

                }
            }
        }

        //Y方向闭操作
        //将相邻的白色区域扩大 连接成一个整体
        element = getStructuringElement(MORPH_RECT, Size(1, 11));
        morphologyEx(absdiffImg, closeImg, MORPH_CLOSE, element);
        //X方向闭操作
        //将相邻的白色区域扩大 连接成一个整体
        element = getStructuringElement(MORPH_RECT, Size(101, 1));
        morphologyEx(closeImg, closeImg, MORPH_CLOSE, element);
        //二值化
        threshold(closeImg, binaryImg, 10, 255, THRESH_BINARY);
   
        //再次颜色过滤
        for (int x = 0; x < hsvImg.rows; x++)
        {
            for (int y = 0; y < hsvImg.cols; y++)
            {
                int H = hsvImg.at<Vec3b>(x, y)[0];
                int S = hsvImg.at<Vec3b>(x, y)[1];
                int V = hsvImg.at<Vec3b>(x, y)[2];
                //蓝色
                if ((H >= 100) && (H <= 120) && (S >= 43) && (V >= 46) && (binaryImg.at<uchar>(x,y) == 255))
                {
                    //binaryImg.at<uchar>(x, y) = 255;

                }
                //黄色
                else if ((H >= 19) && (H <= 26) && (S >= 43) && (V >= 46) && (binaryImg.at<uchar>(x, y) == 255))
                {
                    //binaryImg.at<uchar>(x, y) = 255;

                }
                //绿色
                else if ((H >= 35) && (H <= 77) && (S >= 43) && (V >= 46) && (binaryImg.at<uchar>(x, y) == 255))
                {
                    //binaryImg.at<uchar>(x, y) = 255;

                }

                else
                {
                    binaryImg.at<uchar>(x, y) = 0;
                }
            }
        }

        //Y方向开操作
        element = getStructuringElement(MORPH_RECT, Size(1, 7));
        morphologyEx(binaryImg, openImg, MORPH_CLOSE, element);
        //X方向开操作
        element = getStructuringElement(MORPH_RECT, Size(13, 1));
        morphologyEx(openImg, openImg, MORPH_CLOSE, element);

        //边沿检测
        Canny(openImg, canImg, 25, 125);
        // 进行连通区域检测
        vector<vector<Point>> contours;
        findContours(canImg, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
        int count = 0;
        // 对每个连通区域进行处理
        for (int j = 0; j < contours.size(); j++)
        {
            // 计算连通区域的外接矩形框
            Rect boundingRect = cv::boundingRect(contours[j]);

            // 根据外接矩形框大小和比例进行筛选，以排除非车牌区域
            float aspectRatio = static_cast<float>(boundingRect.width) / static_cast<float>(boundingRect.height);
            if (aspectRatio > 1.25 && aspectRatio < 5 && contours[j].size() > 30)
            {
                
                //提取感兴趣区域
                Mat roiImg;
                roiImg = img(boundingRect);
                Mat shrink;
                //调整大小
                resize(roiImg, shrink, Size(138, 38), 0, 0, INTER_LANCZOS4);
                //灰度化
                Mat roiGray;
                cvtColor(shrink, roiGray, COLOR_BGR2GRAY);
                //提取lbp特征
                Mat lbpImg = Mat(36, 136, CV_8UC1, Scalar(0));
                elbp(roiGray, lbpImg, 1, 8);
                //提取特征向量
                Mat m = getLBPH(lbpImg, 256, 17, 4, false);
                //筛选
                int result;
                result = svmpre->predict(m);

                if (result == 1)
                {

                    count++;
                    //只保留一个
                    if (count == 1)
                    {
                        Mat roiBinary;

                        if (colorJudge(shrink) == 0)
                        {
                            //二值化
                            threshold(roiGray, roiBinary, Otsu(roiGray), 255, THRESH_BINARY);
                        }
                        else if (colorJudge(shrink) == 1 || colorJudge(shrink) == 2)
                        {
                            //取反
                            threshold(roiGray, roiBinary, Otsu(roiGray), 255, THRESH_BINARY_INV);
                        }
                        //自适应阈值边沿检测
                        Mat roiCan;
                        int minthreshold, maxthreshold;
                        float sigma = GetMatMidVal(roiGray);
                        GetMatMinMaxThreshold(roiGray, minthreshold, maxthreshold, sigma);
                        Canny(roiBinary, roiCan, minthreshold, maxthreshold);

                        //霍夫变换
                        float sum = 0;
                        int n = 0;
                        vector<Vec4f> plines;//吧每个像素点的平面坐标转化为极坐标产生的曲线放入集合中
                        HoughLinesP(roiCan, plines, 1, CV_PI / 180.0, 20, 0, 136);
                        for (size_t i = 0; i < plines.size(); i++)
                        {
                            Vec4f hline = plines[i];
                            float theta = atan((hline[1] - hline[3]) / (hline[0] - hline[2]));
                            if (theta<(45 * CV_PI) / 180.0 && theta > -(45 * CV_PI) / 180.0)
                            {
                                sum += theta;
                                n++;
                            }
                            //line(shrink, Point(hline[0], hline[1]), Point(hline[2], hline[3]), Scalar(0, 0, 255), 1, LINE_AA);
                        }
                        float angle = sum / n;
                        //传入中心和角度，得到旋转矩形
                        Mat M = getRotationMatrix2D(Point2f(136 / 2, 36 / 2), angle, 1.0);
                        //旋转后图像
                        Mat roiTrans = Mat(36, 136, CV_8UC1, Scalar(0));
                        //旋转
                        warpAffine(roiBinary, roiTrans, M, Size(136, 36), INTER_LINEAR, 0, Scalar(255, 0, 0));
                        imshow("img", roiTrans);
                        waitKey(0);

                    }
                }
                //// 在原始图像上绘制车牌定位结果
                //rectangle(img, boundingRect, Scalar(0, 0, 255), 2);
            }
        }

        if (count == 0)
        {
            cout << "没有找到目标" << endl;
            imshow("img", img);
            waitKey(0);
        }
        else
        {
            count = 0;
        }
    }
    cv::destroyAllWindows();
}






//大津法计算二值化阈值
double Otsu(Mat& image) 
{

    int threshold = 0;
    double maxVariance = 0;
    double w0 = 0, w1 = 0;//前景与背景像素点所占比例
    double u0 = 0, u1 = 0;//前景与背景像素值平均灰度
    double histogram[256] = { 0 };
    double Num = image.cols * image.rows;
    //统计256个bin，每个bin像素的个数
    for (int i = 0; i < image.rows; i++) {
        const uchar* p = image.ptr<uchar>(i);
        for (int j = 0; j < image.cols; j++) {
            histogram[int(*p++)]++; //cout<<"Histogram[data[i*image.step+j]]++:;"<<histogram[int(*p++)]++<<endl; 
        }
    }
    //前景像素统计
    for (int i = 0; i < 255; i++) {
        w0 = 0;
        w1 = 0;
        u0 = 0;
        u1 = 0;
        for (int j = 0; j <= i; j++) {
            w0 = w0 + histogram[j];//以i为阈值，统计前景像素个数
            u0 = u0 + j * histogram[j];//以i为阈值，统计前景像素灰度总和
        }
        w0 = w0 / Num; u0 = u0 / w0;

        //背景像素统计
        for (int j = i + 1; j <= 255; j++) {
            w1 = w1 + histogram[j];//以i为阈值，统计前景像素个数
            u1 = u1 + j * histogram[j];//以i为阈值，统计前景像素灰度总和
        }
        w1 = w1 / Num; u1 = u1 / w1;
        double variance = w0 * w1 * (u1 - u0) * (u1 - u0); //当前类间方差计算
        if (variance > maxVariance)
        {
            maxVariance = variance;
            threshold = i;
        }
    }
    //cout << "threshold:" << threshold << endl;
    return threshold;
}

//判断车牌颜色
//0:蓝色
//1：黄色
//2：绿色
int colorJudge(Mat img)
{
    Mat hsvImg;

    int blue = 0;
    int yellow = 0;
    int green = 0;

    //转换HSV空间
    cvtColor(img, hsvImg, COLOR_BGR2HSV);
    //再次颜色过滤
    for (int x = 0; x < hsvImg.rows; x++)
    {
        for (int y = 0; y < hsvImg.cols; y++)
        {
            int H = hsvImg.at<Vec3b>(x, y)[0];
            int S = hsvImg.at<Vec3b>(x, y)[1];
            int V = hsvImg.at<Vec3b>(x, y)[2];
            //蓝色
            if ((H >= 100) && (H <= 120) && (S >= 43) && (V >= 46))
            {
                blue++;
            }
            //黄色
            else if ((H >= 19) && (H <= 26) && (S >= 43) && (V >= 46))
            {
                yellow++;
            }
            //绿色
            else if ((H >= 35) && (H <= 77) && (S >= 43) && (V >= 46))
            {
                green++;
            }

        }
    }
    return  blue > yellow ? (blue > green ? 0 : 2) : (yellow > green ? 1 : 2);
}


