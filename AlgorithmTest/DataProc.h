#pragma once

#include <string>
#include <vector>

class DataProc
{
public:
	DataProc();
	~DataProc();
private:
	std::vector<std::string> m_PathVec;		//图像文件路径存放容器
public:
	bool Convert2Gray(std::string path, int w, int h);	//将指定路径下的文件转换为灰度图像
	bool GetImgPath(std::string path);		//由制定的路径获得当前目录下的所有图像文件路径
	bool CheckIsImg(std::string path);		//检查是否为图像文件
	bool ImageConvert(std::string path);	//将固定目录下的图像转换为标定彩色图像和目标图像
	bool ImageCut(std::string src_path, std::string dst_path, int width=500, int hight=500);	//将图片再切分为规定大小的尺寸	
	bool Image2Unet(std::string src_path);	//将图片转换为U-net使用的图片形式
	bool Image2LabelImg(std::string src_path, std::string dst_path);	//将手动划分好的数据只保留肿瘤区域
	bool JPEG2BMP(std::string src_path, std::string dst_path);	//将分割好的图像（GIF格式）转换成jpg格式的图像
};

