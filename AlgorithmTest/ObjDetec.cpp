#include "stdafx.h"
#include "ObjDetec.h"

#define DEBUG

using std::endl;
using std::cout;

cv::Rect img1_rect(266, 146, 87, 87);	//
cv::Rect img2_rect(220, 248, 161, 150);	//
cv::Rect img3_rect(200, 83, 124, 124);	//
cv::Rect img4_rect(238, 120, 76, 76);	//
cv::Rect img5_rect(350, 80, 41, 41);	//
cv::Rect img6_rect(26, 10, 52, 53);	//
cv::Rect img7_rect(204, 347, 44, 44);	//
cv::Rect img8_rect(128, 258, 29, 30);	//
cv::Rect img9_rect(33, 196, 42, 43);	//
cv::Rect img10_rect(381, 207, 24, 24);	//

ObjDetec::ObjDetec()
{
	this->m_rect.push_back(img1_rect);
	this->m_rect.push_back(img2_rect);
	this->m_rect.push_back(img3_rect);
	this->m_rect.push_back(img4_rect);
	this->m_rect.push_back(img5_rect);
	this->m_rect.push_back(img6_rect);
	this->m_rect.push_back(img7_rect);
	this->m_rect.push_back(img8_rect);
	this->m_rect.push_back(img9_rect);
	this->m_rect.push_back(img10_rect);

	this->m_param.push_back(param(1.10, 2));
	this->m_param.push_back(param(1.05, 2));
	this->m_param.push_back(param(1.20, 3));
	this->m_param.push_back(param(1.15, 3));
	this->m_param.push_back(param(1.15, 2));
	this->m_param.push_back(param(1.12, 3));
	this->m_param.push_back(param(1.12, 2));
	this->m_param.push_back(param(1.05, 3));
}


ObjDetec::~ObjDetec()
{
}

//��ѵ��XML·��
void ObjDetec::SetXmlPath(std::string Xml_path)
{
	if (Xml_path.length() <= 0)
	{
		std::cout << "û��ѵ��XML" << endl;
		abort();	//��ֹ
	}
	this->m_XmlPath = Xml_path;
}

//Ŀ���������
bool ObjDetec::Detection(const std::string m_DetecPath)
{
	if (m_DetecPath.length() <= 0)
	{
		std::cout << "���õ�û�м��·��" << endl;
		return false;
	}

	std::vector<double> vec_bestIOU;
	for (int i = 1; i <= 10; ++i)
	{
		std::string pic_path = m_DetecPath + "\\" + std::to_string(i) + ".jpg";	//�õ�ͼ��·��

#ifdef DEBUG
		cout << "��ǰ���·����" << pic_path << endl; 
#endif // DEBUG

		std::vector<double>	vec_iou;

		//�����ͼ��ⷽ���õ���IOU
		vec_iou.push_back(Large_Detection(pic_path, this->m_rect[i-1]));
		//����Сͼ��ⷽ���õ���IOU
		vec_iou.push_back(Small_Detection(pic_path, this->m_rect[i-1]));
		for (int j = 0; j < 8; ++j)
		{
			vec_iou.push_back(Param_Detection(pic_path, this->m_rect[i-1], m_param[j]));
		}
		std::sort(vec_iou.begin(), vec_iou.end());	//����
		vec_bestIOU.push_back(vec_iou[9]);	//������ѵ�IOU

#ifdef DEBUG
		cout << "���IOUΪ��" << vec_iou[9] << endl;
#endif // DEBUG
	}

	double sum_iou=0.0;	//�����ܵ�iou
	for (unsigned int i = 0; i < vec_bestIOU.size(); ++i)
	{
		sum_iou += vec_bestIOU[i];
	}

#ifdef DEBUG
	cout << "�õ���ƽ��IOUΪ��" << (sum_iou/10.0) << endl;
#endif

	return true;
}

//����ͼ����
double ObjDetec::Large_Detection(const std::string pic_path, cv::Rect rect)
{
	cv::CascadeClassifier Mycascade;
	if (!Mycascade.load(this->m_XmlPath))
	{
		cout << "[error] �޷����ؼ����������ļ���" << endl;
		return 0;
	}

	cv::Mat frame_gray = cv::imread(pic_path, 0);//��ȡͼƬ    
	if (!frame_gray.data)
	{
		cout << "[error] û��ͼƬ" << endl;
		return 0;
	}

	std::vector<cv::Rect> pedestrain;
	Mycascade.detectMultiScale(frame_gray, pedestrain, 1.1, 3, 0, cv::Size(100, 100));

	cv::Rect best_rect(0, 0, 0, 0);	//��¼��ѵ�λ��
	float best_diff = 10.0;			//��ѵ�ƥ��ֵ
	cv::Mat template_img = cv::imread("E:\\C++_SQL_program\\footballs\\football.png", 0);
	for (unsigned int i = 0; i < pedestrain.size(); i++)
	{
		//�õ����������ͼ��
		cv::Mat Roi_temp = frame_gray(cv::Rect(pedestrain[i].x, pedestrain[i].y, pedestrain[i].width, pedestrain[i].height));

		float temp_diif = cacSIFTFeatureAndCompare(Roi_temp, template_img, 1000);
		//cout << "ƥ��ȣ�" << temp_diif << endl;
		if (best_diff > temp_diif)
		{
			best_rect = pedestrain[i];
			best_diff = temp_diif;
		}
	}

	double result=0;
	if (pedestrain.size() > 0 && best_diff < 0.025 && best_diff > 0.003)
	{
		result = Calc_IOU(rect, best_rect);
		//����С�����������ķ�Χ����ƫ��������Ҫ�ʶ���С������ľ��ο��С
		for (int i = 9; i >= 1; --i)
		{
			double rate((double)(100-i) / 100.0);
			cv::Rect temp_rect = Resize_Rect(best_rect, rate);
			double value = Calc_IOU(rect, temp_rect);
			if (result < value)
			{
				result = value;
			}
		}
	}

	return result;
}

//���Сͼ����
double ObjDetec::Small_Detection(const std::string pic_path, cv::Rect rect)
{
	cv::CascadeClassifier Mycascade;
	if (!Mycascade.load(this->m_XmlPath))
	{
		cout << "[error] �޷����ؼ����������ļ���" << endl;
		return false;
	}
	cv::Mat image = cv::imread(pic_path, 0);//��ȡͼƬ    
	if (!image.data)
	{
		cout << "[error] û��ͼƬ" << endl;
		return false;
	}

	std::vector<cv::Rect> pedestrain;
	cv::Mat frame_gray = image;

	// 1.1 3
	Mycascade.detectMultiScale(frame_gray, pedestrain, 1.1, 2, 0, cv::Size(20, 20));

	cv::Rect best_rect(0, 0, 0, 0);	//��¼��ѵ�λ��
	float best_diff = 0.0;		//��ѵ�ƥ��ֵ
	cv::Mat template_img = cv::imread("E:\\C++_SQL_program\\footballs\\football.png", 0);
	for (unsigned int i = 0; i < pedestrain.size(); i++)
	{
		//�õ����������ͼ��
		cv::Mat Roi_temp = frame_gray(cv::Rect(pedestrain[i].x, pedestrain[i].y, pedestrain[i].width, pedestrain[i].height));

		float temp_diif = cacSIFTFeatureAndCompare(Roi_temp, template_img, 1000);
		//cout << "ƥ��ȣ�" << temp_diif << endl;
		if (temp_diif != 0 && best_diff < temp_diif)
		{
			best_rect = pedestrain[i];
			best_diff = temp_diif;
		}
	}

	double result = 0;
	if (pedestrain.size() > 0)
	{
		result = Calc_IOU(rect, best_rect);
		//����С�����������ķ�Χ����ƫ��������Ҫ�ʶ���С������ľ��ο��С
		for (int i = 9; i >= 1; --i)
		{
			double rate((double)(100 - i) / 100.0);
			cv::Rect temp_rect = Resize_Rect(best_rect, rate);
			double value = Calc_IOU(rect, temp_rect);
			if (result < value)
			{
				result = value;
			}
		}
	}

	return result;
}

//�������õĲ������м��
double ObjDetec::Param_Detection(const std::string pic_path, cv::Rect rect, param local_param)
{
	cv::CascadeClassifier Mycascade;
	if (!Mycascade.load(this->m_XmlPath))
	{
		cout << "[error] �޷����ؼ����������ļ���" << endl;
		return false;
	}
	cv::Mat image = cv::imread(pic_path, 0);//��ȡͼƬ    
	if (!image.data)
	{
		cout << "[error] û��ͼƬ" << endl;
		return false;
	}

	std::vector<cv::Rect> pedestrain;
	cv::Mat frame_gray = image;

	// 1.1 3
	Mycascade.detectMultiScale(frame_gray, pedestrain, local_param.para1, (int)local_param.para2, 0, cv::Size(20, 20));

	cv::Rect best_rect(0, 0, 0, 0);	//��¼��ѵ�λ��
	float best_diff = 0.0;		//��ѵ�ƥ��ֵ
	cv::Mat template_img = cv::imread("E:\\C++_SQL_program\\footballs\\football.png", 0);
	for (unsigned int i = 0; i < pedestrain.size(); i++)
	{
		//�õ����������ͼ��
		cv::Mat Roi_temp = frame_gray(cv::Rect(pedestrain[i].x, pedestrain[i].y, pedestrain[i].width, pedestrain[i].height));

		float temp_diif = cacSIFTFeatureAndCompare(Roi_temp, template_img, 1000);
		//cout << "ƥ��ȣ�" << temp_diif << endl;
		if (temp_diif != 0 && best_diff < temp_diif)
		{
			best_rect = pedestrain[i];
			best_diff = temp_diif;
		}
	}

	double result = 0;
	if (pedestrain.size() > 0)
	{
		result = Calc_IOU(rect, best_rect);
		//����С�����������ķ�Χ����ƫ��������Ҫ�ʶ���С������ľ��ο��С
		for (int i = 9; i >= 1; --i)
		{
			double rate((double)(100 - i) / 100.0);
			cv::Rect temp_rect = Resize_Rect(best_rect, rate);
			double value = Calc_IOU(rect, temp_rect);
			if (result < value)
			{
				result = value;
			}
		}
	}

	return result;
}

//����ѡ�����ľ��ο�Ĵ�С
cv::Rect ObjDetec::Resize_Rect(cv::Rect rect, float rate)
{ 
	int center_x = rect.x + rect.height/2;
	int center_y = rect.y + rect.width/2;
	int new_width = (int)((double)rect.width*rate);
	int new_height = (int)((double)rect.height*rate);
	int new_x = center_x - new_height / 2;
	int new_y = center_y - new_width / 2;

	return cv::Rect(new_x, new_y, new_width, new_height);
}

//���������ཻ���ε����
double ObjDetec::Intersect_Box_S(const cv::Rect rect1, const cv::Rect rect2)
{
	if (rect1.x > rect2.x + rect2.width)
	{ 
		return 0.0;
	}
	if (rect1.y > rect2.y + rect2.height)
	{ 
		return 0.0;
	}
	if (rect1.x + rect1.width < rect2.x)
	{ 
		return 0.0;
	}
	if (rect1.y + rect1.height < rect2.y) 
	{
		return 0.0;
	}
	float colInt = (float)std::abs(std::min(rect1.x + rect1.width, rect2.x + rect2.width) - std::max(rect1.x, rect2.x));
	float rowInt = (float)std::abs(std::min(rect1.y + rect1.height, rect2.y + rect2.height) - std::max(rect1.y, rect2.y));
	float overlapArea = colInt * rowInt; //�ཻ�����  

	return (double)overlapArea;
}

//����IOU
double ObjDetec::Calc_IOU(const cv::Rect rect1, const cv::Rect rect2)
{
	double intersection = this->Intersect_Box_S(rect1, rect2);
	double area1 = rect1.width*rect1.height;
	double area2 = rect2.width*rect2.height;
	double rate = intersection / (area1 + area2 - intersection); //�ཻ���ı���

	return rate;
}

//SHIFT���������ƥ���
float ObjDetec::cacSIFTFeatureAndCompare(cv::Mat srcImage1, cv::Mat srcImage2, float paraHessian)
{
	CV_Assert(srcImage1.data != NULL && srcImage2.data != NULL);
	// ת��Ϊ�Ҷ�  
	cv::Mat grayMat1 = srcImage1, grayMat2 = srcImage2;
	// ��ʼ��SURF���������  
	cv::SurfFeatureDetector surfDetector(paraHessian);
	cv::SurfDescriptorExtractor surfExtractor;
	// �ؼ��㼰����������������  
	std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
	cv::Mat descriptorMat1, descriptorMat2;
	// ����surf�����ؼ���  
	surfDetector.detect(grayMat1, keyPoints1);
	surfDetector.detect(grayMat2, keyPoints2);
	// ����surf������������  
	surfExtractor.compute(grayMat1, keyPoints1, descriptorMat1);
	surfExtractor.compute(grayMat2, keyPoints2, descriptorMat2);
	float result = 0;
	// ������ƥ��  
	if (keyPoints1.size() > 0 && keyPoints2.size() > 0)
	{
		// ��������ƥ���  
		cv::FlannBasedMatcher matcher;
		std::vector< cv::DMatch > matches;
		std::vector<cv::DMatch> viewMatches;
		matcher.match(descriptorMat1, descriptorMat2, matches);
		// ����ƥ���ж�  
		double minDist = 100;
		for (unsigned int i = 0; i < matches.size(); i++)
		{
			if (matches[i].distance < minDist)
				minDist = matches[i].distance;
		}
		// ���������������Ҫ���������  
		int num = 0;
		//std::cout << "minDist: " << minDist << std::endl;
		for (unsigned int i = 0; i < matches.size(); i++)
		{
			// ������ƥ������ж�  
			if (matches[i].distance <= 2 * minDist)
			{
				result += matches[i].distance * matches[i].distance;
				viewMatches.push_back(matches[i]);
				num++;
			}
		}
		// ƥ��ȼ���  
		result /= num;
	}
	return result;
}