// AlgorithmTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include "DataProc.h"
#include "ObjDetec.h"

using std::cout;
using std::endl;

//根据提供的XML去检测目标图像是否包含需要的结果
bool ObjectDetectShow(std::string xml_path, std::string testpic_path);
//负样本被统计为正样本的比例
void CalcReacllRate(std::string path, std::string xml_path);
//感知哈希算法计算相似度度量
int Perceptual_Hash(cv::Mat& img1, cv::Mat& img2);
//SHIFT特征点计算匹配度
float cacSIFTFeatureAndCompare(cv::Mat srcImage1, cv::Mat srcImage2, float paraHessian);
//使用大小不同的窗口进行检测
bool JudgeFunc(std::string xml_path, std::string testpic_path);

int _tmain(int argc, _TCHAR* argv[])
{
	system("color f0");
	DataProc m_datapro;
	//m_datapro.Convert2Gray("E:\\C++_SQL_program\\footballs\\positive", 30, 30);
	//m_datapro.ImageConvert("C:\\Users\\SUCKER\\Desktop\\SegmentationClass\\1");
	//m_datapro.ImageCut("C:\\Users\\SUCKER\\Desktop\\SegmentationClass\\1\\truth_ground", "C:\\Users\\SUCKER\\Desktop\\SegmentationClass\\1\\truth_ground\\500-500", 500, 500);
	//m_datapro.Image2Unet("C:\\Users\\SUCKER\\Desktop\\SegmentationClass\\1\\truth_ground\\500-500");
	//m_datapro.Image2LabelImg("C:\\Users\\SUCKER\\Desktop\\新建文件夹\\Segmentation", "C:\\Users\\SUCKER\\Desktop\\新建文件夹\\Segmentation");
	//m_datapro.JPEG2BMP("C:\\Users\\SUCKER\\Desktop\\seg_case2\\BMP", "C:\\Users\\SUCKER\\Desktop\\seg_case2\\BMP1");

	std::string xml_path = "E:\\C++_SQL_program\\footballs\\negative\\output\\out.xml";
	std::string neg_path = "E:\\C++_SQL_program\\footballs\\Course Project Pack\\dataset1";
	std::string testpic_path = "E:\\C++_SQL_program\\footballs\\Course Project Pack\\dataset1\\999.jpg";
	std::string testpic_path2 = "E:\\C++_SQL_program\\footballs\\Course Project Pack\\dataset2";

	ObjDetec objd;
	objd.SetXmlPath(xml_path);
	objd.Detection(testpic_path2);

	//负样本被统计为正样本的比例
	//CalcReacllRate(neg_path, xml_path);

	//JudgeFunc(xml_path, testpic_path2);
	//ObjectDetectShow(xml_path, testpic_path2);

	system("pause");
	return 0;
}

//根据提供的XML去检测目标图像是否包含需要的结果
bool ObjectDetectShow(std::string xml_path, std::string testpic_path)
{
	cv::CascadeClassifier Mycascade;
	if (!Mycascade.load(xml_path)) 
	{ 
		cout << "[error] 无法加载级联分类器文件！" << endl;
		return false; 
	}
	cv::Mat image = cv::imread(testpic_path, 0);//读取图片    
	if (!image.data)
	{ 
		cout << "[error] 没有图片" << endl;
		return false; 
	}

	std::vector<cv::Rect> pedestrain;
	cv::Mat frame_gray = image;
	//cv::Mat frame_gray(image.size(), CV_8U);
	//cv::cvtColor(image, frame_gray, CV_BGR2GRAY);
	//equalizeHist(frame_gray, frame_gray);

	// 1.1 3
	Mycascade.detectMultiScale(frame_gray, pedestrain, 1.1, 2, 0, cv::Size(20, 20));

	cv::Rect best_rect(0,0,0,0);	//记录最佳的位置
	float best_diff = 0.0;		//最佳的匹配值
	cv::Mat template_img = cv::imread("E:\\C++_SQL_program\\footballs\\football.png",0);
	for (unsigned int i = 0; i < pedestrain.size(); i++)  
	{
		//得到疑似区域的图像
		cv::Mat Roi_temp = frame_gray(cv::Rect(pedestrain[i].x, pedestrain[i].y, pedestrain[i].width, pedestrain[i].height));
		//感知哈希算法计算相似度度量
		//float temp_diif = Perceptual_Hash(Roi_temp, template_img);
		float temp_diif = cacSIFTFeatureAndCompare(Roi_temp, template_img, 1000);
		cout << "匹配度：" << temp_diif << endl;
		if (temp_diif!=0 && best_diff<temp_diif)
		{
			best_rect = pedestrain[i];
			best_diff = temp_diif;
		}
	}

	rectangle(image,                //图像.    
		best_rect,
		cv::Scalar(0, 255, 0),		//线条颜色 (RGB) 或亮度（灰度图像 ）(grayscale image）    
		1);							//组成矩形的线条的粗细程度。取负值时（如 CV_FILLED）函数绘制填充了色彩的矩形 

	cv::imshow("result", image);
	cv::waitKey(0);

	return true;
}

//负样本被统计为正样本的比例
void CalcReacllRate(std::string path, std::string xml_path)
{
	if (path.length() <= 0)
	{
		cout << "输入路径为空" << endl;
		return ;
	}

	cv::CascadeClassifier Mycascade;
	if (!Mycascade.load(xml_path))
	{
		cout << "[error] 无法加载级联分类器文件！" << endl;
		return;
	}
	
	int count = 999, error_count=0;;
	for (int i = 0; i <= count; ++i)
	{
		std::string testpic_path = path + "\\" + std::to_string(i) + ".jpg";
		cv::Mat frame_gray = cv::imread(testpic_path, 0);//读取图片    
		if (!frame_gray.data)
		{
			cout << "[error] 没有图片" << endl;
			return;
		}

		std::vector<cv::Rect> pedestrain;
		Mycascade.detectMultiScale(frame_gray, pedestrain, 1.1, 3, 0, cv::Size(100, 100));

		cv::Rect best_rect(0, 0, 0, 0);	//记录最佳的位置
		float best_diff = 10.0;		//最佳的匹配值
		cv::Mat template_img = cv::imread("E:\\C++_SQL_program\\footballs\\football.png", 0);
		for (unsigned int i = 0; i < pedestrain.size(); i++)
		{
			//得到疑似区域的图像
			cv::Mat Roi_temp = frame_gray(cv::Rect(pedestrain[i].x, pedestrain[i].y, pedestrain[i].width, pedestrain[i].height));

			float temp_diif = cacSIFTFeatureAndCompare(Roi_temp, template_img, 1000);
			//cout << "匹配度：" << temp_diif << endl;
			if (best_diff > temp_diif)
			{
				best_rect = pedestrain[i];
				best_diff = temp_diif;
			}
		}
		if (pedestrain.size() > 0 && best_diff < 0.025 && best_diff > 0.003)
		{
			cout << testpic_path << endl;
			error_count++;
		}
	}

	cout << "负样本被分类为正样本的比例为：" << (double)error_count / (double)count << endl;
}

//使用大小不同的窗口进行检测
bool JudgeFunc(std::string xml_path, std::string testpic_path)
{
	cv::CascadeClassifier Mycascade;
	if (!Mycascade.load(xml_path))
	{
		cout << "[error] 无法加载级联分类器文件！" << endl;
		return false;
	}
	cv::Mat image = cv::imread(testpic_path);//读取图片    
	if (!image.data)
	{
		cout << "[error] 没有图片" << endl;
		return false;
	}

	std::vector<cv::Rect> pedestrain1, pedestrain2;	//保存检测的结果
	cv::Mat frame_gray;
	if (image.type() ==CV_8UC3)
	{
		cv::cvtColor(image, frame_gray, CV_BGR2GRAY);
	}
	else if (image.type() == CV_8UC1)
	{
		frame_gray = image;
	}

	Mycascade.detectMultiScale(frame_gray, pedestrain1, 1.1, 3, 0, cv::Size(20, 20));	//小窗口
	Mycascade.detectMultiScale(frame_gray, pedestrain2, 1.1, 3, 0, cv::Size(100, 100));	//大窗口

	cv::Rect best_rect(0, 0, 0, 0);	//记录最佳的位置
	float best_diff = 0.0;			//最佳的匹配值
	cv::Mat template_img = cv::imread("E:\\C++_SQL_program\\footballs\\1212.jpg", 0);	//加载模板图像
	int result_count=0;	//检测结果记录

	//小窗口图像
	for (unsigned int i = 0; i < pedestrain1.size(); i++)
	{
		//得到疑似区域的图像
		cv::Mat Roi_temp = frame_gray(cv::Rect(pedestrain1[i].x, pedestrain1[i].y, pedestrain1[i].width, pedestrain1[i].height));
		float temp_diif = cacSIFTFeatureAndCompare(Roi_temp, template_img, 1000);
		if (temp_diif != 0 && best_diff < temp_diif)
		{
			best_rect = pedestrain1[i];
			best_diff = temp_diif;
		}
	}
	if (pedestrain1.size() > 0)
	{
		rectangle(image, best_rect, cv::Scalar(0, 0, 255), 2);	//对于疑似区域进行绘制 
		result_count++;
	}

	//大窗口图像
	best_diff = 10.0;		//最佳的匹配值
	for (unsigned int i = 0; i < pedestrain2.size(); i++)
	{
		//得到疑似区域的图像
		cv::Mat Roi_temp = frame_gray(cv::Rect(pedestrain2[i].x, pedestrain2[i].y, pedestrain2[i].width, pedestrain2[i].height));
		float temp_diif = cacSIFTFeatureAndCompare(Roi_temp, template_img, 1000);
		//cout << "匹配度：" << temp_diif << endl;
		if (best_diff > temp_diif)
		{
			best_rect = pedestrain2[i];
			best_diff = temp_diif;
		}
	}
	if (pedestrain2.size() > 0 && best_diff < 0.025 && best_diff > 0.003)
	{
		rectangle(image, best_rect, cv::Scalar(0, 255, 0), 2);	//对于疑似区域进行绘制
		result_count++;
	}

	cv::imshow("result", image);
	cv::waitKey(0);
	if (result_count > 0)
	{
		return true;
	}

	return false;
}

//SHIFT特征点计算匹配度
float cacSIFTFeatureAndCompare(cv::Mat srcImage1, cv::Mat srcImage2, float paraHessian)
{
	CV_Assert(srcImage1.data != NULL && srcImage2.data != NULL);
	// 转换为灰度  
	cv::Mat grayMat1=srcImage1, grayMat2=srcImage2;
	//cv::cvtColor(srcImage1, grayMat1, CV_RGB2GRAY);
	//cv::cvtColor(srcImage2, grayMat2, CV_RGB2GRAY);
	// 初始化SURF检测描述子  
	cv::SurfFeatureDetector surfDetector(paraHessian);
	cv::SurfDescriptorExtractor surfExtractor;
	// 关键点及特征描述矩阵声明  
	std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
	cv::Mat descriptorMat1, descriptorMat2;
	// 计算surf特征关键点  
	surfDetector.detect(grayMat1, keyPoints1);
	surfDetector.detect(grayMat2, keyPoints2);
	// 计算surf特征描述矩阵  
	surfExtractor.compute(grayMat1, keyPoints1, descriptorMat1);
	surfExtractor.compute(grayMat2, keyPoints2, descriptorMat2);
	float result = 0;
	// 特征点匹配  
	if (keyPoints1.size() > 0 && keyPoints2.size() > 0)
	{
		// 计算特征匹配点  
		cv::FlannBasedMatcher matcher;
		std::vector< cv::DMatch > matches;
		std::vector<cv::DMatch> viewMatches;
		matcher.match(descriptorMat1, descriptorMat2, matches);
		// 最优匹配判断  
		double minDist = 100;
		for (unsigned int i = 0; i < matches.size(); i++)
		{
			if (matches[i].distance < minDist)
				minDist = matches[i].distance;
		}
		// 计算距离特征符合要求的特征点  
		int num = 0;
		//std::cout << "minDist: " << minDist << std::endl;
		for (unsigned int i = 0; i < matches.size(); i++)
		{
			// 特征点匹配距离判断  
			if (matches[i].distance <= 2 * minDist)
			{
				result += matches[i].distance * matches[i].distance;
				viewMatches.push_back(matches[i]);
				num++;
			}
		}
		// 匹配度计算  
		result /= num;
	}
	return result;
}

//感知哈希算法计算相似度度量
int Perceptual_Hash(cv::Mat& img1, cv::Mat& img2)
{
	cv::Mat matSrc1 = img1, matDst1;
	cv::Mat matSrc2 = img2, matDst2;
	cv::resize(matSrc1, matDst1, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
	cv::resize(matSrc2, matDst2, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);

	//cv::cvtColor(matDst1, matDst1, CV_BGR2GRAY);
	//cv::cvtColor(matDst2, matDst2, CV_BGR2GRAY);

	int iAvg1 = 0, iAvg2 = 0;
	int arr1[64], arr2[64];

	for (int i = 0; i < 8; i++)
	{
		uchar* data1 = matDst1.ptr<uchar>(i);
		uchar* data2 = matDst2.ptr<uchar>(i);

		int tmp = i * 8;

		for (int j = 0; j < 8; j++)
		{
			int tmp1 = tmp + j;

			arr1[tmp1] = data1[j] / 4 * 4;
			arr2[tmp1] = data2[j] / 4 * 4;

			iAvg1 += arr1[tmp1];
			iAvg2 += arr2[tmp1];
		}
	}

	iAvg1 /= 64;
	iAvg2 /= 64;

	for (int i = 0; i < 64; i++)
	{
		arr1[i] = (arr1[i] >= iAvg1) ? 1 : 0;
		arr2[i] = (arr2[i] >= iAvg2) ? 1 : 0;
	}

	int iDiffNum = 0;

	for (int i = 0; i < 64; i++)
	if (arr1[i] != arr2[i])
		++iDiffNum;

	return iDiffNum;
}
