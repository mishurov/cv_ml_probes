#include <iostream>
#include <vector>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

using namespace cv;
using namespace tensorflow;


int g_heightIn = 360;
int g_widthIn = 480;
int g_depthIn = 3;

int g_heightOut = 224;
int g_widthOut = 224;

int g_border = 25;
int g_gap = 35;
int g_pairGap = 10;
int g_numRows = 3;

float g_videoScale = 1.7;

bool g_showGrid = false;

unsigned char g_bgColor[] = {255, 255, 255};

unsigned char g_colorMap[][3] = {
    { 128, 128, 128 },
    {  64,   0, 128 },
    { 192, 192, 128 },
    { 128,  64, 128 },
    {  60,  40, 222 },
    { 128, 128,   0 },
    { 192, 128, 128 },
    {  64,  64, 128 },
    { 128,   0,   0 },
    {  64,  64,   0 },
    {   0, 128, 192 },
    {   0,   0,   0 }
};

std::string g_modelsDirectory = "../training/export/";

std::vector<std::string> g_dataset = {
	"../training/CamVid/train/0016E5_06390.png",
	"../training/CamVid/train/0016E5_05520.png",
	"../training/CamVid/train/0006R0_f01830.png",
	"../training/CamVid/test/Seq05VD_f03300.png",
	"../training/CamVid/val/0016E5_08045.png",
	"../training/CamVid/val/0016E5_08153.png"
};

std::string g_modelPath = "";

std::string g_windowName = "SegNet";

std::string g_inputKey = "image";


Tensor matToImgTensor(Mat& mat)
{
	mat.convertTo(mat, CV_32FC3, 1.0 / 255.0);
	Tensor tensor(DT_FLOAT, TensorShape({1, g_heightOut, g_widthOut, g_depthIn}));
	auto tensorMapped = tensor.tensor<float, 4>();

	for (int y = 0; y < g_heightOut; ++y) {
	    const float* sourceRow = ((float*)mat.data) + (y * g_widthOut * 3);
	    for (int x = 0; x < g_widthOut; ++x) {
		const float* sourcePixel = sourceRow + (x * 3);
		tensorMapped(0, y, x, 0) = sourcePixel[2];
		tensorMapped(0, y, x, 1) = sourcePixel[1];
		tensorMapped(0, y, x, 2) = sourcePixel[0];
	    }
	}

	return tensor;
}


Mat mapColors(const Mat &img)
{
	Mat matC(g_widthOut, g_heightOut, CV_8UC3);
	for(int i = 0; i < img.rows; i++) {
		for(int j = 0; j < img.cols; ++j) {
			int index = img.at<int>(i, j);
			matC.at<Vec3b>(i, j) = Vec3b(g_colorMap[index]);
		}
	}
	return matC;
}


Mat tensorToMat(Tensor& tensor)
{
	int *p = tensor.flat<int32>().data();
	Mat matP(g_widthOut, g_heightOut, CV_32SC1, p);
	return mapColors(matP);
}


void cropResize(Mat& img)
{
	int minDim = min(g_heightIn, g_widthIn);
	int cropX = (g_widthIn - minDim) / 2;
	int cropY = (g_heightIn - minDim) / 2;
	auto roi = Rect(cropX, cropY, g_widthIn - cropX, g_heightIn - cropY);
	img = img(roi);
	Size size(g_heightOut, g_widthOut);
	resize(img, img, size);
}


Point getCanvasSize()
{
  int numImages = g_dataset.size();
  int numRows = min(g_numRows, numImages);
  int w = numRows * g_widthOut * 2;
  w += numRows * g_pairGap;
  w += (numRows - 1) * g_gap;
  w += g_border * 2;

  int numCols = (numImages - 1) / g_numRows + 1;
  int h = numCols * g_heightOut;
  h += (numCols - 1) * g_gap;
  h += g_border * 2;
  return Point(w, h);
}


Mat processImage(SavedModelBundle& bundle, Mat img) {
	Tensor tensor = matToImgTensor(img);
	std::vector<std::pair<string, Tensor> > inputs = {
		std::make_pair(g_inputKey, tensor)
	};
	std::vector<Tensor> outputs;
	Status status = bundle.session->Run(
		inputs,
		{"predict/preds:0"},
		{},
		&outputs
	);
	Tensor output = outputs.at(0);
	return tensorToMat(output);
}


void showGrid(SavedModelBundle& bundle)
{
	Mat canvas(
		getCanvasSize(),
		CV_8UC3,
		Vec3b(g_bgColor)
	);
	Mat gap(g_heightOut, g_pairGap, CV_8UC3, Vec3b(g_bgColor));
	int pairWidth = g_widthOut * 2 + g_pairGap;
	Mat img;
	
	for (int i = 0; i < g_dataset.size(); i++) {
		img = imread(g_dataset.at(i));
		cvtColor(img, img, CV_BGR2RGB);
		cropResize(img);
		
		std::cout << "processing test image " << g_dataset.at(i) << std::endl;
		Mat seg = processImage(bundle, img);

		int n = i / g_numRows;
		int m = i - n * g_numRows;
		int offsetX = g_border + m * pairWidth + m * g_gap;
		int offsetY = g_border + n * g_heightOut + n * g_gap;
		Mat pair;
		Mat imgGap;
		hconcat(img, gap, imgGap);
		hconcat(imgGap, seg, pair);

		Rect roi(Point(offsetX, offsetY), pair.size());
		pair.copyTo(canvas(roi));
	}

	namedWindow(g_windowName, WINDOW_AUTOSIZE);
	imshow(g_windowName, canvas);
	waitKey(0);
}


void showVideo(SavedModelBundle& bundle)
{
	int scaledWidth = (int)((float)g_widthOut * g_videoScale);
	int scaledHeight = (int)((float)g_heightOut * g_videoScale);
	Size scaledSize(scaledWidth, scaledHeight);
	Mat gap(scaledHeight, g_pairGap, CV_8UC3, Vec3b(g_bgColor));
	Mat img;
	Mat imgGap;
	Mat seg;
	Mat pair;
	double currentPos;
	String path;
	int i = 0;
	while (true)
	{
		path = format("../training/CamVid/test/Seq05VD_f0%04d.png", i * 30);
		i++;
		img = imread(path);
		cvtColor(img, img, CV_BGR2RGB);
		cropResize(img);

		seg = processImage(bundle, img);
		resize(img, img, scaledSize);
		resize(seg, seg, scaledSize);
		hconcat(img, gap, imgGap);
		hconcat(imgGap, seg, pair);

		imshow(g_windowName, pair);
		if (waitKey(1) == 27) break;
	}
}


std::string getFirstFile(const std::string& path)
{
	DIR *pdir = NULL;
	pdir = opendir(path.c_str());

	std::string filename;
	std::string dot = ".";
	std::string ddot = "..";

	struct dirent *pent = NULL;

	if (pdir == NULL) {
		std::cout << "pdir error";
		exit(3);
	}
	while (pent = readdir(pdir))
	{
		if (pent == NULL) {
			std::cout << "pent error";
			exit(3);
		}
		
		if (dot.compare(pent->d_name) && ddot.compare(pent->d_name)) {
			filename = pent->d_name;
			break;
		}
	}
	closedir(pdir);

	return filename;
}

void parseArguments(int argc, char** argv) {
	const String keys =
		"{ m model  | | path to the model file }"
		"{ g grid   | | show grid instead of video }"
	;

	CommandLineParser parser = CommandLineParser(argc, argv, keys);
	parser.about("hello");

	if (parser.has("model"))
		g_modelPath = parser.get<std::string>("model");

	if (parser.has("grid"))
		g_showGrid = true;
}


int main(int argc, char** argv)
{
	parseArguments(argc, argv);

	std::string modelPath;
	if (g_modelPath.empty()) {
		std::string filename = getFirstFile(g_modelsDirectory);
		if (filename.empty()) {
			std::cout << "Model not found, exiting\n";
			exit(3);
		}
		modelPath = g_modelsDirectory + filename;
	} else {
		modelPath = g_modelPath;
	}

	RunOptions runOptions;
	SessionOptions sessionOptions;

	SavedModelBundle bundle;
	LoadSavedModel(sessionOptions, runOptions, modelPath,
		{kSavedModelTagServe}, &bundle);

	if (g_showGrid)
		showGrid(bundle);
	else
		showVideo(bundle);
	return 0;
}
