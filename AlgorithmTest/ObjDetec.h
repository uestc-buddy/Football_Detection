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

//�����ṹ��
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
	std::string m_XmlPath;			//������������·��
	std::vector<cv::Rect> m_rect;	//��׼������
	std::vector<param> m_param;		//�����ṹ

public:
	void SetXmlPath(std::string Xml_path);				//����ѵ��XML·��
	bool Detection(const std::string m_DetecPath);		//Ŀ���������
	double Large_Detection(const std::string pic_path, cv::Rect rect);	//����ͼ����
	double Small_Detection(const std::string pic_path, cv::Rect rect);	//���Сͼ����
	double Param_Detection(const std::string pic_path, cv::Rect rect, param local_param);	//�������õĲ������м��
	double Intersect_Box_S(const cv::Rect rect1, const cv::Rect rect2);	//���������ཻ���ε����
	cv::Rect Resize_Rect(cv::Rect rect, float rate);				//����ѡ�����ľ��ο�Ĵ�С
	double Calc_IOU(const cv::Rect rect1, const cv::Rect rect2);	//����IOU
	float cacSIFTFeatureAndCompare(cv::Mat srcImage1, cv::Mat srcImage2, float paraHessian);//SHIFT���������ƥ���
};

