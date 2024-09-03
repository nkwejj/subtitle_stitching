#include<iostream>
#include<vector>
#include<cstring>
#include<map>

#include<opencv2/opencv.hpp>
#include"opencv2/features2d.hpp"

using namespace cv;
using namespace std;


void show(Mat src) {
	//显示图片
	if (src.empty()) {
		cout << "could not load image...\n";
		return;
	}
	imshow("image", src);
	waitKey(0);
	destroyAllWindows();
}

void print(Mat src) {
	//std::cout << src1.rows << " " << src1.cols << std::endl;
	//std::cout << src2.rows << " " << src2.cols << std::endl;

	//std::cout << src1.size()<<" "<<src1.channels()<<"\n";

	//打印矩阵内容
	for (int r = 0; r < src.rows; r++) {
		const int* ptr = src.ptr<int>(r);
		for (int c = 0; c < src.cols; c++) {
			cout << unsigned int(ptr[c]) << ",";
		}
		cout << endl;
	}
}

void draw_feature(Mat image, vector<KeyPoint> kps, Mat descriptors) {
	//绘制特征点
	Mat output;
	drawKeypoints(image, kps, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SIFT Keypoints", output);
	waitKey(0);
}

void delete_black(Mat& image) {
	//去除右侧黑边

	int rs = image.rows;
	int cs = image.cols;

	try {
		// 转换为灰度图像
		Mat grayImage;
		cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

		// 阈值处理
		Mat binaryImage;
		threshold(grayImage, binaryImage, 1, 255, cv::THRESH_BINARY);

		// 使用boundingRect确定黑边范围
		Rect boundingRect = cv::boundingRect(binaryImage);

		// // 在原始图像上绘制边界矩形
		//rectangle(image, boundingRect, cv::Scalar(0, 255, 0), 2);

		image = image(boundingRect);
		//show(image);

		//cout << rs << " " << cs << endl;
		//cout << image.rows << " " << image.cols << endl;
	}
	catch (Exception e) {
		cout << e.err;
	}

}


void mystitch(string left, string right, Rect roi, bool processed = false) {
	string output_roll = "result/roll.jpg";
	string output_fix = "result/fix.jpg";

	// [预处理]
	Mat src1 = imread(left);
	Mat src2 = imread(right);

	/*
	* 提取 ROI 区域
	* x y width height
	*/
	//获取ROI区域，这是要拼接的滚动字幕区域
	Mat image1 = src2(roi);//右图
	Mat image2;//左图
	if (processed) {
		image2 = src1;
	}
	else {
		//如果传入的是未处理的原始图像，直接赋值
		//（为了能够多次拼接）
		image2 = src1(roi);
	}

	imwrite("result/right.jpg", image1);
	imwrite("result/left.jpg", image2);
	// [预处理]

	// [SIFT]
	//检测特征点	
	Ptr<SIFT> detector = SIFT::create();
	vector<KeyPoint> kps1, kps2;
	detector->detect(image1, kps1);
	detector->detect(image2, kps2);

	//计算特征描述符
	Mat descriptors1;
	Mat descriptors2;
	detector->compute(image1, kps1, descriptors1);
	detector->compute(image2, kps2, descriptors2);
	//绘制特征点
	//draw_feature(image1, kps1, descriptors1);
	//draw_feature(image2, kps2, descriptors2);
	// [SIFT]

	// [BruteForce Matcher]
	//创建特征点匹配器
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	//进行特征匹配
	/*
	class DMatch{
		...
		int queryIdx; // 特征点在第一幅图像中的索引
		int trainIdx; // 特征点在第二幅图像中的索引
		float distance; // 两个特征点之间的距离
	};
	*/
	vector<vector<DMatch>> matches;
	vector<DMatch> good_matches;
	//matcher->match(descriptors1, descriptors2, matches);//vector<DMatch> matches
	matcher->knnMatch(descriptors1, descriptors2, matches, 2);//k=2

	// Lowe's algorithm,获取优秀匹配点
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance > 0.8 * matches[i][1].distance)
		{
			//最近距离与第二最近距离的比率，大于 0.8，则忽略它们
			//消除了大约 90％的错误匹配，而同时只去除了 5％的正确匹配
		}
		else {
			good_matches.push_back(matches[i][0]);
		}
	}

	//绘制匹配结果
	//Mat output;
	//drawMatches(image1, kps1, image2, kps2, matches, output,
	//	Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//imshow("Feature Matches", output);
	//waitKey(0);
	// [BruteForce Matcher]

	// [RANSAC]
	//使用 RANSAC 算法估计透视变换矩阵
	map<double, int> count;//L1距离 出现次数

	vector<Point2f> srcPoints, dstPoints;
	for (int i = 0; i < good_matches.size(); i++) {
		srcPoints.push_back(kps1[good_matches[i].queryIdx].pt);
		dstPoints.push_back(kps2[good_matches[i].trainIdx].pt);
		int x1 = kps1[good_matches[i].queryIdx].pt.x;
		int y1 = kps1[good_matches[i].queryIdx].pt.y;

		int x2 = kps2[good_matches[i].trainIdx].pt.x;
		int y2 = kps2[good_matches[i].trainIdx].pt.y;

		int L1 = abs(x1 - x2) + abs(y1 - y2);

		if (count.find(L1) == count.end()) {
			//key不存在
			count.insert(pair<double, int>(L1, 1));
		}
		else {
			map<double, int>::iterator it = count.find(L1);
			it->second = it->second + 1;
		}
	}

	vector< pair<double, int> > vec;
	for (map<double, int>::iterator it = count.begin(); it != count.end(); it++) {
		vec.push_back(pair<double, int>(it->first, it->second));
	}
	sort(vec.begin(), vec.end(), [](pair<double, int>a, pair<double, int>b) {return a.second > b.second; });

	///*==========判断是否为滚动字幕===========*/
	//if (vec.size() == 0 || vec[0].second < 60) {
	//	//根据相似点匹配数量进行判断
	//	cout << endl;
	//	cout << "=================================" << endl;
	//	cout << "           不是滚动字幕" << endl;
	//	cout << "=================================" << endl;
	//	Mat res;
	//	vconcat(image2, image1, res);
	//	imwrite(output_fix, res);
	//	return;
	//}
	//else {
	//	cout << endl;
	//	cout << "=================================" << endl;
	//	cout << "           是滚动字幕" << endl;
	//	cout << "=================================" << endl;
	//}

	//图像配准
	/*透视变换矩阵:
	* 这个矩阵可以用于将源图像中的内容映射到目标图像的坐标系中，
	* 从而实现图像拼接、图像配准等功能
	*/
	//计算单应性矩阵，用于透视校正或图像对齐
	Mat H = findHomography(srcPoints, dstPoints, RANSAC);
	//cout << H << endl << endl;

	//对图像进行透视变换
	Mat imageTransform1, imageTransform2;
	//warpPerspective(image1, imageTransform1, H, Size(image1.cols + image2.cols, image2.rows));
	warpPerspective(image1, imageTransform1, H, Size(image1.cols + image2.cols, image2.rows));
	//show(imageTransform1);

	//图像拷贝
	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
	int dst_height = image2.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	image2.copyTo(dst(Rect(0, 0, image2.cols, image2.rows)));//复制到左边

	//show(dst);

	delete_black(dst);//去除右侧黑边

	//show(dst);
	imwrite(output_roll, dst);

	// [RANSAC]

}


int main(int argc, char** argv) {
	//{
	//	//测试1：滚动字幕
	//	string left = "Sample1/016660.jpg";//height: 576 width: 720
	//	string right = "Sample1/016665.jpg";//height: 576 width: 720

	//	Rect label(165, 520, 550, 50);//滚动字幕，sample1  [x y width height]
	//	mystitch(left, right, label);//181 个 goog_matches

	//}
	//{
	//	//测试2：固定字幕
	//	string left = "mysample/202405010251.png";//height: 378 width: 720
	//	string right = "mysample/202405010311.png";//height: 378 width: 720

	//	Rect label(152, 337, 565, 38);//固定字幕

	//	mystitch(left, right, label);//137 个 goog_matches
	//}
	//{
	//	//测试3：多个滚动字幕拼接
	//	string left1 = "Sample1/016660.jpg";//height: 576 width: 720
	//	string right1 = "Sample1/016665.jpg";//height: 576 width: 720

	//	Rect label(165, 520, 550, 50);//滚动字幕，sample1

	//	mystitch(left1, right1, label);//181 个 goog_matches

	//	string left2 = "result/roll.jpg";
	//	string right2 = "Sample1/016717.jpg";//height: 576 width: 720

	//	mystitch(left2, right2, label, true);
	//}
	//{
	//	//测试4：距离很近的滚动字幕，判断为 是滚动字幕
	//	string left = "Sample1/016660.jpg";//height: 576 width: 720
	//	string right = "Sample1/016661.jpg";//height: 576 width: 720

	//	Rect label(165, 520, 550, 50);//滚动字幕，sample1
	//	mystitch(left, right, label);
	//}
	//{
	//	//测试5：距离很远的滚动字幕，判断为 不是滚动字幕
	//	string left = "Sample1/016660.jpg";//height: 576 width: 720
	//	string right = "Sample1/016886.jpg";//height: 576 width: 720

	//	Rect label(165, 520, 550, 50);//sample1

	//	mystitch(left, right, label);
	//}
	//{
	//	//测试6：两个不同帧但相同的固定字幕
	//	string left = "Sample2/016660.jpg";//height: 480 width: 640
	//	string right = "Sample2/016665.jpg";//height: 480 width: 640

	//	Rect label(90, 376, 400, 24);//固定字幕，sample2
	//	
	//	mystitch(left, right, label);
	//}

	//{
	//	//Sample1
	//	// 016660.jpg - 016908.jpg
	//	int count = 16660;
	//	string left = "Sample1/0" + to_string(count) + ".jpg";//height: 576 width: 720
	//	count += 8;
	//	string right = "Sample1/0" + to_string(count) + ".jpg";//height: 576 width: 720
	//	count += 40;

	//	Rect label(165, 520, 530, 50);//滚动字幕

	//	mystitch(left, right, label);

	//	while (count <= 16908) {
	//		left = "result/roll.jpg";
	//		right = "Sample1/0" + to_string(count) + ".jpg";
	//		count += 40;
	//		mystitch(left, right, label, true);
	//	}
	//	
	//}
	
	//{
	//	//Sample2
	//	//016660.jpg - 016840.jpg
	//	int count = 16660;
	//	string left = "Sample2/0" + to_string(count) + ".jpg";//height: 480 width: 640
	//	count += 30;
	//	string right = "Sample2/0" + to_string(count) + ".jpg";//height: 480 width: 640
	//	count += 30;

	//	Rect label(150, 430, 480, 25);//滚动字幕

	//	mystitch(left, right, label);

	//	while (count <= 16840) {
	//		left = "result/roll.jpg";
	//		right = "Sample2/0" + to_string(count) + ".jpg";
	//		count += 30;
	//		mystitch(left, right, label, true);
	//	}

	//}

	return 0;
}
