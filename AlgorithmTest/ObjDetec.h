#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\objdetect\objdetect.hpp>

//参数结构体
typedef struct Param
{
	double para1;
	double para2;
	Param(double a, double b)
	{
		para1 = a;
		para2 = b;
	}
} param;

class ObjDetec
{
public:
	ObjDetec();
	~ObjDetec();
private:
	std::string m_XmlPath;			//级联分类器的路径
	std::vector<cv::Rect> m_rect;	//标准的区域
	std::vector<param> m_param;		//参数结构

public:
	void SetXmlPath(std::string Xml_path);				//设置训练XML路径
	bool Detection(const std::string m_DetecPath);		//目标检测总入口
	double Large_Detection(const std::string pic_path, cv::Rect rect);	//检测大图像函数
	double Small_Detection(const std::string pic_path, cv::Rect rect);	//检测小图像函数
	double Param_Detection(const std::string pic_path, cv::Rect rect, param local_param);	//根据设置的参数进行检测
	double Intersect_Box_S(const cv::Rect rect1, const cv::Rect rect2);	//计算两个相交矩形的面积
	cv::Rect Resize_Rect(cv::Rect rect, float rate);				//重置选出来的矩形框的大小
	double Calc_IOU(const cv::Rect rect1, const cv::Rect rect2);	//计算IOU
	float cacSIFTFeatureAndCompare(cv::Mat srcImage1, cv::Mat srcImage2, float paraHessian);//SHIFT特征点计算匹配度
};

