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

//��ָ��·���µ��ļ�ת��Ϊ�Ҷ�ͼ��
bool DataProc::Convert2Gray(std::string path, int w, int h)
{
	//��ȡ�ļ��������е�ͼ���ļ���
	if (!this->GetImgPath(path))
	{
		cout << "ת��ʧ��" << endl;
		return false;
	}

	//����Ƿ���ͼ��
	if (this->m_PathVec.size() == 0)
	{
		cout << "��ǰĿ¼��û��ָ����ʽ��ͼ���ļ�" << endl;
		return false;
	}

	std::string coutput_path = path + "\\output";
	if (_access(coutput_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(coutput_path.c_str()))
		{
			cout << "�������Ŀ¼ʧ��" << endl;
			return false;
		}
	}


	unsigned int vec_size(this->m_PathVec.size());
	for (unsigned int i = 0; i < vec_size; ++i)
	{
		cv::Mat img = cv::imread(this->m_PathVec[i], 0);
		int rows = img.rows;
		int cols = img.cols;
		if (rows != w || cols != h)	//����ͼ�񵽹涨�ĳߴ�
		{
			cv::resize(img, img, cv::Size(w, h), (0, 0), (0, 0), cv::INTER_LINEAR);
		}
		cv::imwrite(coutput_path+"\\img" + std::to_string(i)+".png", img);
	}

	return true;
}

//���ƶ���·����õ�ǰĿ¼�µ�����ͼ���ļ�·��
bool DataProc::GetImgPath(std::string path)
{
	//��Ч���ж�
	if (path.length() == 0)
	{
		cout << "�����ļ�����Ч" << endl;
		return false;
	}
	//�ļ���
	std::vector<std::string> file_name;

	//�ļ����  
	long hFile = 0;
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮  
			//�������,�����б�  
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

//����Ƿ�Ϊͼ���ļ�
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
		cout << "ͼ���ʽ����֧�֣�" << path << endl;

	return false;
}

//���̶�Ŀ¼�µ�ͼ��ת��Ϊ�궨��ɫͼ���Ŀ��ͼ��
bool DataProc::ImageConvert(std::string path)
{
	if ("" == path)
	{
		cout << "path error" << endl;
		return false;
	}

	//��ȡ�ļ��������е�ͼ���ļ���
	if (!this->GetImgPath(path))
	{
		cout << "ת��ʧ��" << endl;
		return false;
	}

	//����Ƿ���ͼ��
	if (this->m_PathVec.size() == 0)
	{
		cout << "��ǰĿ¼��û��ָ����ʽ��ͼ���ļ�" << endl;
		return false;
	}

	std::string trueground_path = path + "\\truth_ground";
	if (_access(trueground_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(trueground_path.c_str()))
		{
			cout << "�������Ŀ¼ʧ��" << endl;
			return false;
		}
	}
	std::string seg_path = path + "\\Seg";
	if (_access(seg_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(seg_path.c_str()))
		{
			cout << "�������Ŀ¼ʧ��" << endl;
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

		cv::Mat trueground_img(rows, cols, CV_8UC3, cv::Scalar::all(0));	//truth groundͼ��
		cv::Mat Seg_img(rows, cols, CV_8UC1, cv::Scalar::all(0));			//�ָ�ͼ��
		std::string img_name = this->m_PathVec[i].substr(this->m_PathVec[i].find_last_of("\\")+1, 6);

		for (int m=0; m<rows; ++m)
		{
			data = Seg_img.ptr<unsigned char>(m);
			for (int n = 0; n < cols; ++n)
			{
				int blue = img.at<cv::Vec3b>(m, n)[0];
				int green = img.at<cv::Vec3b>(m, n)[1];
				int red = img.at<cv::Vec3b>(m, n)[2];
				if (abs(red - 255) <= 10 && blue < 30 && green < 30)	//������֯=1
				{
					trueground_img.at<cv::Vec3b>(m, n)[2] = 255;
					*data++ = 1;
				}
				else if (abs(blue - 255) <= 10 && red < 30 && green < 30)	//��������֯=2
				{
					trueground_img.at<cv::Vec3b>(m, n)[0] = 255;
					*data++ = 2;
				}
				else if (abs(green - 255) <= 10 && blue < 30 && red < 30)	//������֯=3
				{
					trueground_img.at<cv::Vec3b>(m, n)[1] = 255;
					*data++ = 3;
				}
				else	//����ͼ
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

//��ͼƬ���з�Ϊ�涨��С�ĳߴ�
bool DataProc::ImageCut(std::string src_path, std::string dst_path, int width, int hight)
{
	if ("" == src_path)
	{
		cout << "path error" << endl;
		return false;
	}

	//��ȡ�ļ��������е�ͼ���ļ���
	if (!this->GetImgPath(src_path))
	{
		cout << "ת��ʧ��" << endl;
		return false;
	}

	//����Ƿ���ͼ��
	if (this->m_PathVec.size() == 0)
	{
		cout << "��ǰĿ¼��û��ָ����ʽ��ͼ���ļ�" << endl;
		return false;
	}

	std::string trueground_path = dst_path;
	if (_access(trueground_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(trueground_path.c_str()))
		{
			cout << "�������Ŀ¼ʧ��" << endl;
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

//��ͼƬת��ΪU-netʹ�õ�ͼƬ��ʽ
bool DataProc::Image2Unet(std::string src_path)
{
	if ("" == src_path)
	{
		cout << "path error" << endl;
		return false;
	}

	//��ȡ�ļ��������е�ͼ���ļ���
	if (!this->GetImgPath(src_path))
	{
		cout << "ת��ʧ��" << endl;
		return false;
	}

	//����Ƿ���ͼ��
	if (this->m_PathVec.size() == 0)
	{
		cout << "��ǰĿ¼��û��ָ����ʽ��ͼ���ļ�" << endl;
		return false;
	}

	std::string unet_path = src_path + "\\unet";
	if (_access(unet_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(unet_path.c_str()))
		{
			cout << "�������Ŀ¼ʧ��" << endl;
			return false;
		}
	}
	std::string seg_path = src_path + "\\Seg";
	if (_access(seg_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(seg_path.c_str()))
		{
			cout << "�������Ŀ¼ʧ��" << endl;
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

		cv::Mat trueground_img(rows, cols, CV_8UC3, cv::Scalar::all(0));	//truth groundͼ��
		cv::Mat Seg_img(rows, cols, CV_8UC1, cv::Scalar::all(0));			//�ָ�ͼ��
		cv::Mat unet_img(rows, cols, CV_8UC1, cv::Scalar::all(0));			//unetͼ��
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
				if (abs(red - 255) <= 10 && blue < 30 && green < 30)	//������֯=1
				{
					trueground_img.at<cv::Vec3b>(m, n)[2] = 255;
					*data++ = 1;
					*data_u = 1;
				}
				else if (abs(blue - 255) <= 10 && red < 30 && green < 30)	//��������֯=2
				{
					trueground_img.at<cv::Vec3b>(m, n)[0] = 255;
					*data++ = 2;
				}
				else if (abs(green - 255) <= 10 && blue < 30 && red < 30)	//������֯=3
				{
					trueground_img.at<cv::Vec3b>(m, n)[1] = 255;
					*data++ = 3;
				}
				else	//����ͼ
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

//���ֶ����ֺõ�����ֻ������������
bool DataProc::Image2LabelImg(std::string src_path, std::string dst_path)
{
	if ("" == src_path)
	{
		cout << "path error" << endl;
		return false;
	}

	//��ȡ�ļ��������е�ͼ���ļ���
	if (!this->GetImgPath(src_path))
	{
		cout << "ת��ʧ��" << endl;
		return false;
	}

	//����Ƿ���ͼ��
	if (this->m_PathVec.size() == 0)
	{
		cout << "��ǰĿ¼��û��ָ����ʽ��ͼ���ļ�" << endl;
		return false;
	}

	dst_path = dst_path + "\\output";
	if (_access(dst_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(dst_path.c_str()))
		{
			cout << "�������Ŀ¼ʧ��" << endl;
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

		cv::Mat dst_img(rows, cols, CV_8UC1, cv::Scalar::all(0));			//�ָ�ͼ��
		std::string img_name = this->m_PathVec[i].substr(this->m_PathVec[i].find_last_of("\\") + 1, 6);

		for (int m = 0; m < rows; ++m)
		{
			data = dst_img.ptr<unsigned char>(m);
			for (int n = 0; n < cols; ++n)
			{
				int blue = img.at<cv::Vec3b>(m, n)[0];
				int green = img.at<cv::Vec3b>(m, n)[1];
				int red = img.at<cv::Vec3b>(m, n)[2];
				if (abs(red - 255) <= 10 && blue < 30 && green < 30)	//������֯=1
				{
					*data++ = 255;
				}
				else	//����ͼ
				{
					*data++ = 0;
				}
			}
		}

		cv::imwrite(dst_path + "\\" + img_name + ".png", dst_img);
	}
	return true;
}

//���ָ�õ�ͼ��GIF��ʽ��ת����jpg��ʽ��ͼ��
bool DataProc::JPEG2BMP(std::string src_path, std::string dst_path)	
{
	if ("" == src_path)
	{
		cout << "path error" << endl;
		return false;
	}

	//��ȡ�ļ��������е�ͼ���ļ���
	if (!this->GetImgPath(src_path))
	{
		cout << "ת��ʧ��" << endl;
		return false;
	}

	//����Ƿ���ͼ��
	if (this->m_PathVec.size() == 0)
	{
		cout << "��ǰĿ¼��û��ָ����ʽ��ͼ���ļ�" << endl;
		return false;
	}

	if (_access(dst_path.c_str(), 0) == -1)
	{
		if (-1 == _mkdir(dst_path.c_str()))
		{
			cout << "�������Ŀ¼ʧ��" << endl;
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
