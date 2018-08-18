#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GL/glu.h>

#include <opencv2/opencv.hpp>


cv::Mat_<float> getCameraMatrix(float fovY, float aspect, float w, float h);
cv::Matx44f getProjectionMatrix(cv::Mat_<float> cam);
GLuint compileShader(GLuint type, const std::string& src);

void initBgGlData();
void initBgTexture(cv::Mat& img);
void drawBg(cv::Mat* img);

