#pragma once

#include <string>
#include <vector>

class DataProc
{
public:
	DataProc();
	~DataProc();
private:
	std::vector<std::string> m_PathVec;		//ͼ���ļ�·���������
public:
	bool Convert2Gray(std::string path, int w, int h);	//��ָ��·���µ��ļ�ת��Ϊ�Ҷ�ͼ��
	bool GetImgPath(std::string path);		//���ƶ���·����õ�ǰĿ¼�µ�����ͼ���ļ�·��
	bool CheckIsImg(std::string path);		//����Ƿ�Ϊͼ���ļ�
	bool ImageConvert(std::string path);	//���̶�Ŀ¼�µ�ͼ��ת��Ϊ�궨��ɫͼ���Ŀ��ͼ��
	bool ImageCut(std::string src_path, std::string dst_path, int width=500, int hight=500);	//��ͼƬ���з�Ϊ�涨��С�ĳߴ�	
	bool Image2Unet(std::string src_path);	//��ͼƬת��ΪU-netʹ�õ�ͼƬ��ʽ
	bool Image2LabelImg(std::string src_path, std::string dst_path);	//���ֶ����ֺõ�����ֻ������������
	bool JPEG2BMP(std::string src_path, std::string dst_path);	//���ָ�õ�ͼ��GIF��ʽ��ת����jpg��ʽ��ͼ��
};

