#pragma once
#include <opencv2/opencv.hpp>  
#include<iostream>  
#include<stdio.h>


using namespace std;
using namespace cv;

//��Mat����λ��
int GetMatMidVal(Mat& img);
//������Ӧ��ֵ����С�����ֵ
void GetMatMinMaxThreshold(Mat& img, int& minval, int& maxval, float sigma);







