#pragma once
#include <opencv2/opencv.hpp>  
#include<iostream>  
#include<stdio.h>


using namespace std;
using namespace cv;

//求Mat的中位数
int GetMatMidVal(Mat& img);
//求自适应阈值的最小和最大值
void GetMatMinMaxThreshold(Mat& img, int& minval, int& maxval, float sigma);







