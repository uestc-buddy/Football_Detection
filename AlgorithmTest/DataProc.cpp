#include "stdafx.h"
#include "DataProc.h"
#include <string>
#include <io.h>
#include <direct.h> 
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using std::endl;
using std::cout;

DataProc::DataProc()
{
}


DataProc::~DataProc()
{
}

//将指定路径下的文件转换为灰度图像
bool DataProc::Convert2Gray(std::string path, int w, int h)
{
	//获取文件夹下所有的图像文件名
	if (!this->GetImgPath(path))
	{
		cout << "转换失败" << endl;
		return false;
	}

	//检查是否有图像
	if (this->m_PathVec.size() == 0)
	{
		cout << "当前目录下没有指定格式的图像文件" << endl;
		return false;
	}

	std::string coutput_path = path + "\\output";
	if (_access(coutput_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(coutput_path.c_str()))
		{
			cout << "创建输出目录失败" << endl;
			return false;
		}
	}


	unsigned int vec_size(this->m_PathVec.size());
	for (unsigned int i = 0; i < vec_size; ++i)
	{
		cv::Mat img = cv::imread(this->m_PathVec[i], 0);
		int rows = img.rows;
		int cols = img.cols;
		if (rows != w || cols != h)	//调整图像到规定的尺寸
		{
			cv::resize(img, img, cv::Size(w, h), (0, 0), (0, 0), cv::INTER_LINEAR);
		}
		cv::imwrite(coutput_path+"\\img" + std::to_string(i)+".png", img);
	}

	return true;
}

//由制定的路径获得当前目录下的所有图像文件路径
bool DataProc::GetImgPath(std::string path)
{
	//有效性判断
	if (path.length() == 0)
	{
		cout << "输入文件名无效" << endl;
		return false;
	}
	//文件名
	std::vector<std::string> file_name;

	//文件句柄  
	long hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				//if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					//getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				file_name.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	unsigned int vec_size(file_name.size());
	for (unsigned int i = 0; i < vec_size; ++i)
	{
		std::string temp = file_name[i];
		if (this->CheckIsImg(temp))
			this->m_PathVec.push_back(temp);
	}

	return true;
}

//检查是否为图像文件
inline bool DataProc::CheckIsImg(std::string path)
{
	if (path.length() == 0)
		return false;
	else if ((int)path.find(".bmp") > 0 || (int)path.find(".BMP") > 0)
	{
		return true;
	}
	else if ((int)path.find(".png") > 0 || (int)path.find(".PNG") > 0)
	{
		return true;
	}
	else if ((int)path.find(".jpg") > 0 || (int)path.find(".JPG") > 0)
	{
		return true;
	}
	else if ((int)path.find(".jpeg") > 0 || (int)path.find(".JPEG") > 0)
	{
		return true;
	}
	else if ((int)path.find(".gif") > 0 || (int)path.find(".GIF") > 0)
	{
		return true;
	}
	else
		cout << "图像格式不被支持：" << path << endl;

	return false;
}

//将固定目录下的图像转换为标定彩色图像和目标图像
bool DataProc::ImageConvert(std::string path)
{
	if ("" == path)
	{
		cout << "path error" << endl;
		return false;
	}

	//获取文件夹下所有的图像文件名
	if (!this->GetImgPath(path))
	{
		cout << "转换失败" << endl;
		return false;
	}

	//检查是否有图像
	if (this->m_PathVec.size() == 0)
	{
		cout << "当前目录下没有指定格式的图像文件" << endl;
		return false;
	}

	std::string trueground_path = path + "\\truth_ground";
	if (_access(trueground_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(trueground_path.c_str()))
		{
			cout << "创建输出目录失败" << endl;
			return false;
		}
	}
	std::string seg_path = path + "\\Seg";
	if (_access(seg_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(seg_path.c_str()))
		{
			cout << "创建输出目录失败" << endl;
			return false;
		}
	}

	unsigned int vec_size(this->m_PathVec.size());
	for (unsigned int i = 0; i < vec_size; ++i)
	{
		cv::Mat img = cv::imread(this->m_PathVec[i], CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH);
		int rows = img.rows;
		int cols = img.cols;
		int channles =img.channels();
		unsigned char* data = nullptr;

		cv::Mat trueground_img(rows, cols, CV_8UC3, cv::Scalar::all(0));	//truth ground图像
		cv::Mat Seg_img(rows, cols, CV_8UC1, cv::Scalar::all(0));			//分割图像
		std::string img_name = this->m_PathVec[i].substr(this->m_PathVec[i].find_last_of("\\")+1, 6);

		for (int m=0; m<rows; ++m)
		{
			data = Seg_img.ptr<unsigned char>(m);
			for (int n = 0; n < cols; ++n)
			{
				int blue = img.at<cv::Vec3b>(m, n)[0];
				int green = img.at<cv::Vec3b>(m, n)[1];
				int red = img.at<cv::Vec3b>(m, n)[2];
				if (abs(red - 255) <= 10 && blue < 30 && green < 30)	//肿瘤组织=1
				{
					trueground_img.at<cv::Vec3b>(m, n)[2] = 255;
					*data++ = 1;
				}
				else if (abs(blue - 255) <= 10 && red < 30 && green < 30)	//正常脑组织=2
				{
					trueground_img.at<cv::Vec3b>(m, n)[0] = 255;
					*data++ = 2;
				}
				else if (abs(green - 255) <= 10 && blue < 30 && red < 30)	//其它组织=3
				{
					trueground_img.at<cv::Vec3b>(m, n)[1] = 255;
					*data++ = 3;
				}
				else	//背景图
				{
					*data++ = 0;
				}
			}
		}

		cv::imwrite(trueground_path + "\\" + img_name + ".jpg", trueground_img);
		//cv::imwrite(seg_path + "\\" + img_name + ".png", Seg_img);
	}

	return true;
}

//将图片再切分为规定大小的尺寸
bool DataProc::ImageCut(std::string src_path, std::string dst_path, int width, int hight)
{
	if ("" == src_path)
	{
		cout << "path error" << endl;
		return false;
	}

	//获取文件夹下所有的图像文件名
	if (!this->GetImgPath(src_path))
	{
		cout << "转换失败" << endl;
		return false;
	}

	//检查是否有图像
	if (this->m_PathVec.size() == 0)
	{
		cout << "当前目录下没有指定格式的图像文件" << endl;
		return false;
	}

	std::string trueground_path = dst_path;
	if (_access(trueground_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(trueground_path.c_str()))
		{
			cout << "创建输出目录失败" << endl;
			return false;
		}
	}

	unsigned int vec_size(this->m_PathVec.size());
	int img_index=0;
	std::string name_temp = "000000";
	for (unsigned int i = 0; i < vec_size; ++i)
	{
		cv::Mat img = cv::imread(this->m_PathVec[i], CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH);
		//cv::Mat dst1, dst2, dst3;
		//cv::medianBlur(img, img, 3);
		//cv::medianBlur(img, dst2, 5);
		//cv::blur(img, dst3, cv::Size(3,3));
		int rows(img.rows);
		int cols(img.cols);
		for (int m = 0; m < rows - width; ++m)
		{
			for (int n = 0; n < cols - hight; ++n)
			{
				cv::Mat temp = img(cv::Rect(m,n,width,hight));
				std::string save_name = std::to_string(img_index++);
				save_name = name_temp.substr(0, name_temp.length()-save_name.length())+save_name;
				cv::imwrite(trueground_path+"\\"+save_name+".jpg", temp);
			}
		}
	}

	return true;
}

//将图片转换为U-net使用的图片形式
bool DataProc::Image2Unet(std::string src_path)
{
	if ("" == src_path)
	{
		cout << "path error" << endl;
		return false;
	}

	//获取文件夹下所有的图像文件名
	if (!this->GetImgPath(src_path))
	{
		cout << "转换失败" << endl;
		return false;
	}

	//检查是否有图像
	if (this->m_PathVec.size() == 0)
	{
		cout << "当前目录下没有指定格式的图像文件" << endl;
		return false;
	}

	std::string unet_path = src_path + "\\unet";
	if (_access(unet_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(unet_path.c_str()))
		{
			cout << "创建输出目录失败" << endl;
			return false;
		}
	}
	std::string seg_path = src_path + "\\Seg";
	if (_access(seg_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(seg_path.c_str()))
		{
			cout << "创建输出目录失败" << endl;
			return false;
		}
	}

	unsigned int vec_size(this->m_PathVec.size());
	for (unsigned int i = 0; i < vec_size; ++i)
	{
		cv::Mat img = cv::imread(this->m_PathVec[i], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		int rows = img.rows;
		int cols = img.cols;
		int channles = img.channels();
		unsigned char* data = nullptr;
		unsigned char* data_u = nullptr;

		cv::Mat trueground_img(rows, cols, CV_8UC3, cv::Scalar::all(0));	//truth ground图像
		cv::Mat Seg_img(rows, cols, CV_8UC1, cv::Scalar::all(0));			//分割图像
		cv::Mat unet_img(rows, cols, CV_8UC1, cv::Scalar::all(0));			//unet图像
		std::string img_name = this->m_PathVec[i].substr(this->m_PathVec[i].find_last_of("\\") + 1, 6);

		for (int m = 0; m < rows; ++m)
		{
			data = Seg_img.ptr<unsigned char>(m);
			data_u = unet_img.ptr<unsigned char>(m);
			for (int n = 0; n < cols; ++n)
			{
				int blue = img.at<cv::Vec3b>(m, n)[0];
				int green = img.at<cv::Vec3b>(m, n)[1];
				int red = img.at<cv::Vec3b>(m, n)[2];
				if (abs(red - 255) <= 10 && blue < 30 && green < 30)	//肿瘤组织=1
				{
					trueground_img.at<cv::Vec3b>(m, n)[2] = 255;
					*data++ = 1;
					*data_u = 1;
				}
				else if (abs(blue - 255) <= 10 && red < 30 && green < 30)	//正常脑组织=2
				{
					trueground_img.at<cv::Vec3b>(m, n)[0] = 255;
					*data++ = 2;
				}
				else if (abs(green - 255) <= 10 && blue < 30 && red < 30)	//其它组织=3
				{
					trueground_img.at<cv::Vec3b>(m, n)[1] = 255;
					*data++ = 3;
				}
				else	//背景图
				{
					*data++ = 0;
				}
				data_u++;
			}
		}

		cv::imwrite(unet_path + "\\" + img_name + ".png", unet_img);
		//cv::imwrite(seg_path + "\\" + img_name + ".png", Seg_img);
	}

	return true;
}

//将手动划分好的数据只保留肿瘤区域
bool DataProc::Image2LabelImg(std::string src_path, std::string dst_path)
{
	if ("" == src_path)
	{
		cout << "path error" << endl;
		return false;
	}

	//获取文件夹下所有的图像文件名
	if (!this->GetImgPath(src_path))
	{
		cout << "转换失败" << endl;
		return false;
	}

	//检查是否有图像
	if (this->m_PathVec.size() == 0)
	{
		cout << "当前目录下没有指定格式的图像文件" << endl;
		return false;
	}

	dst_path = dst_path + "\\output";
	if (_access(dst_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(dst_path.c_str()))
		{
			cout << "创建输出目录失败" << endl;
			return false;
		}
	}

	unsigned int vec_size(this->m_PathVec.size());
	for (unsigned int i = 0; i < vec_size; ++i)
	{
		cv::Mat img = cv::imread(this->m_PathVec[i], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		int rows = img.rows;
		int cols = img.cols;
		int channles = img.channels();
		unsigned char* data = nullptr;

		cv::Mat dst_img(rows, cols, CV_8UC1, cv::Scalar::all(0));			//分割图像
		std::string img_name = this->m_PathVec[i].substr(this->m_PathVec[i].find_last_of("\\") + 1, 6);

		for (int m = 0; m < rows; ++m)
		{
			data = dst_img.ptr<unsigned char>(m);
			for (int n = 0; n < cols; ++n)
			{
				int blue = img.at<cv::Vec3b>(m, n)[0];
				int green = img.at<cv::Vec3b>(m, n)[1];
				int red = img.at<cv::Vec3b>(m, n)[2];
				if (abs(red - 255) <= 10 && blue < 30 && green < 30)	//肿瘤组织=1
				{
					*data++ = 255;
				}
				else	//背景图
				{
					*data++ = 0;
				}
			}
		}

		cv::imwrite(dst_path + "\\" + img_name + ".png", dst_img);
	}
	return true;
}

//将分割好的图像（GIF格式）转换成jpg格式的图像
bool DataProc::JPEG2BMP(std::string src_path, std::string dst_path)	
{
	if ("" == src_path)
	{
		cout << "path error" << endl;
		return false;
	}

	//获取文件夹下所有的图像文件名
	if (!this->GetImgPath(src_path))
	{
		cout << "转换失败" << endl;
		return false;
	}

	//检查是否有图像
	if (this->m_PathVec.size() == 0)
	{
		cout << "当前目录下没有指定格式的图像文件" << endl;
		return false;
	}

	if (_access(dst_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(dst_path.c_str()))
		{
			cout << "创建输出目录失败" << endl;
			return false;
		}
	}

	unsigned int vec_size(this->m_PathVec.size());
	for (unsigned int i = 0; i < vec_size; ++i)
	{
		std::string img_path = this->m_PathVec[i];
		if (img_path.find_first_of(".jpg") > 0)
		{
			cv::Mat img = cv::imread(this->m_PathVec[i], 0);
			std::string img_name = this->m_PathVec[i].substr(this->m_PathVec[i].find_last_of("\\") + 1, 6);
			std::string save_path = dst_path + "\\" + img_name;// + ".bmp";
			cv::imwrite(save_path, img);
		}
	}

	return true;
}
