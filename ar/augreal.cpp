#include "common.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/tracking.hpp>


using namespace cv;

std::string g_videoPath = "../../samples/matrix.mp4";
bool g_selectRoi = false;
float g_fovY = 0.9;
float g_aspect = 2.1;

Ptr<FeatureDetector> g_detector = xfeatures2d::SIFT::create(1000);
Ptr<DescriptorExtractor> g_extractor = g_detector;
Ptr<DescriptorMatcher> g_matcher = BFMatcher::create(NORM_L2);

std::string g_windowName = "SIFT";


std::string g_vertexShaderMeshSrc =
R"(
#version 130

in vec4 position;
in vec3 vertexColor;
out vec3 fragmentColor;

void main()
{
	gl_Position = gl_ModelViewProjectionMatrix * position;
	fragmentColor = vertexColor * (0.75 - position.z);
}
)";


std::string g_fragmentShaderMeshSrc =
R"(
#version 130
in vec3 fragmentColor;

void main()
{
	gl_FragColor = vec4(fragmentColor, 1.0f);
}
)";


GLfloat g_width  = 0.3;
GLfloat g_height = 0.3;
GLfloat g_depth  = 0.3;

float g_vertCoordsMesh[] = {
	 g_width, g_height, g_depth,
	-g_width, g_height, g_depth,
	-g_width,-g_height, g_depth,
	 g_width,-g_height, g_depth,

	 g_width, g_height, g_depth,
	 g_width,-g_height, g_depth,
	 g_width,-g_height,-g_depth,
	 g_width, g_height,-g_depth,

	 g_width, g_height, g_depth,
	 g_width, g_height,-g_depth,
	-g_width, g_height,-g_depth,
	-g_width, g_height, g_depth,

	-g_width,-g_height,-g_depth,
	-g_width, g_height,-g_depth,
	 g_width, g_height,-g_depth,
	 g_width,-g_height,-g_depth,

	-g_width,-g_height,-g_depth,
	-g_width,-g_height, g_depth,
	-g_width, g_height, g_depth,
	-g_width, g_height,-g_depth,

	-g_width,-g_height,-g_depth,
	 g_width,-g_height,-g_depth,
	 g_width,-g_height, g_depth,
	-g_width,-g_height, g_depth
};


float g_vertColsMesh[] = {
	1,0,0,  1,0,0,  1,0,0,  1,0,0,
	0,1,0,  0,1,0,  0,1,0,  0,1,0,
	0,0,1,  0,0,1,  0,0,1,  0,0,1,
	1,0,0,  1,0,0,  1,0,0,  1,0,0,
	0,1,0,  0,1,0,  0,1,0,  0,1,0,
	0,0,1,  0,0,1,  0,0,1,  0,0,1,
};


GLuint g_programMesh;
GLuint g_vertCoordBufMesh;
GLuint g_vertColBufMesh;

std::vector<KeyPoint> g_queryKp;
Mat g_queryDes;
std::vector<cv::DMatch> g_matches;
std::vector<std::vector<cv::DMatch> > g_knnMatches;

Mat g_warpedImg;
Mat g_grayImg;
Mat g_roughHomography, g_refinedHomography, g_homography;
std::vector<cv::Point2f> g_points2d;

Mat g_raux, g_taux;
Matx44f g_projectionMat;


struct Pattern
{
	Size size;
	std::vector<KeyPoint> kp;
	Mat des;
	std::vector<cv::Point2f> p2d;
	std::vector<Point3f> p3d;
};

Pattern g_pattern;


void initGL()
{
	GLenum error = glewInit();
	if (error != GLEW_OK) {
		std::cout << "Failed to initialize glew\n";
		return;
	}
	char * versionGL = (char *)(glGetString(GL_VERSION));

	initBgGlData();

	GLuint vertexShaderMesh = compileShader(
		GL_VERTEX_SHADER, g_vertexShaderMeshSrc
	);
	GLuint fragmentShaderMesh = compileShader(
		GL_FRAGMENT_SHADER, g_fragmentShaderMeshSrc
	);

	g_programMesh = glCreateProgram();
	glAttachShader(g_programMesh, vertexShaderMesh);
	glAttachShader(g_programMesh, fragmentShaderMesh);
	glLinkProgram(g_programMesh);


	glGenBuffers(1, &g_vertCoordBufMesh);
	glBindBuffer(GL_ARRAY_BUFFER, g_vertCoordBufMesh);

	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertCoordsMesh),
			g_vertCoordsMesh, GL_STATIC_DRAW);

	glGenBuffers(1, &g_vertColBufMesh);
	glBindBuffer(GL_ARRAY_BUFFER, g_vertColBufMesh);

	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertColsMesh),
			g_vertColsMesh, GL_STATIC_DRAW);
}


void drawMesh()
{
	glUseProgram(g_programMesh);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, g_vertCoordBufMesh);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, g_vertColBufMesh);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDrawArrays(GL_QUADS, 0, 24);
	glDisable(GL_DEPTH_TEST);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	//glUseProgram(0);
}


Matx44f getModelViewMatrix()
{
	// initial rotation of the cube
	Mat preRvec(3, 1, CV_32F, { 0.5, 0.1, 0.3 });

	Mat_<float> tvec;
	g_taux.convertTo(tvec, CV_32F);
	Mat rvec(3, 1, CV_32F);
	g_raux.convertTo(rvec, CV_32F);

	rvec = preRvec + rvec;

	Mat_<float> rmat(3,3); 
	Rodrigues(rvec, rmat);

	rmat = rmat.t();
	Matx44f mv = Matx44f::eye();

	for (int col = 0; col < 3; col++) {
		for (int row = 0; row < 3; row++)
			mv(row, col) = -rmat(row, col);
		mv(3, col) = -tvec(col);
	}

	return mv;
}


void drawGL(void* param)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	Mat* img = (Mat*)param;
	drawBg(img);

	if (g_raux.empty()) return;
	
	Matx44f modelViewMat = getModelViewMatrix();
 
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(g_projectionMat.val);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glLoadMatrixf(modelViewMat.val);

	drawMesh();
}


void extractFeatures(const Mat& img, std::vector<KeyPoint>& kp, Mat& des)
{
	g_detector->detect(img, kp);
	g_extractor->compute(img, kp, des);
}


void initPattern(const Mat& frame)
{
	Mat img = frame.clone();

	if (g_selectRoi) {
		Rect2d roi = selectROI(g_windowName, img);
		img = frame(roi);
	}

	g_pattern.size = Size(img.cols, img.rows);
	g_pattern.p2d.resize(4);
	g_pattern.p3d.resize(4);

	float w = img.rows;
	float h = img.cols;

	float maxDim = std::max(w, h);
	float unitW = w / maxDim;
	float unitH = h / maxDim;

	g_pattern.p2d[0] = Point2f(0, 0);
	g_pattern.p2d[1] = Point2f(w, 0);
	g_pattern.p2d[2] = Point2f(w, h);
	g_pattern.p2d[3] = Point2f(0, h);

	g_pattern.p3d[0] = Point3f(-unitW, -unitH, 0);
	g_pattern.p3d[1] = Point3f( unitW, -unitH, 0);
	g_pattern.p3d[2] = Point3f( unitW,  unitH, 0);
	g_pattern.p3d[3] = Point3f(-unitW,  unitH, 0);

	cvtColor(img, g_grayImg, CV_BGR2GRAY);
	extractFeatures(g_grayImg, g_pattern.kp, g_pattern.des);

	g_matcher->clear();
	std::vector<Mat> descriptors(1);
	descriptors.at(0) = g_pattern.des.clone(); 
	g_matcher->add(descriptors);
	g_matcher->train();
}


bool refineMatches(const std::vector<KeyPoint>& kp,
			std::vector<DMatch>& matches, Mat& homography)
{
	int minMatches = 8;

	if (matches.size() < minMatches)
		return false;

	std::vector<Point2f> srcPoints(matches.size());
	std::vector<Point2f> dstPoints(matches.size());

	for (int i = 0; i < matches.size(); i++) {
		srcPoints[i] = g_pattern.kp[matches[i].trainIdx].pt;
		dstPoints[i] = kp[matches[i].queryIdx].pt;
	}

	std::vector<unsigned char> mask(srcPoints.size());
	homography = findHomography(srcPoints, dstPoints, CV_FM_RANSAC, 3, mask);

	std::vector<cv::DMatch> inliers;
	for (int i = 0; i < mask.size(); i++) {
		if (mask[i])
			inliers.push_back(matches[i]);
	}

	matches.swap(inliers);
	return matches.size() > minMatches;
}


void getMatches(std::vector<cv::DMatch>& matches)
{
	matches.clear();

	g_knnMatches.clear();
	g_matcher->knnMatch(g_queryDes, g_knnMatches, 2);

	float minRatio = 1.0 / 1.5;

	for (auto &n : g_knnMatches) {
		DMatch bestMatch   = n[0];
		DMatch betterMatch = n[1];

		float distanceRatio = bestMatch.distance / betterMatch.distance;
		if (distanceRatio < minRatio) {
			matches.push_back(bestMatch);
		}
	}
}


bool findPattern(const Mat& img)
{
	cvtColor(img, g_grayImg, CV_BGR2GRAY);
	extractFeatures(g_grayImg, g_queryKp, g_queryDes);
	getMatches(g_matches);
	bool homographyFound = refineMatches(
		g_queryKp, g_matches, g_roughHomography
	);

	if (homographyFound) {
		warpPerspective(g_grayImg, g_warpedImg, g_roughHomography,
				g_pattern.size, WARP_INVERSE_MAP|INTER_CUBIC);
		std::vector<KeyPoint> warpedKp;
		std::vector<DMatch> refinedMatches;

		extractFeatures(g_warpedImg, warpedKp, g_queryDes);
		getMatches(refinedMatches);

		homographyFound = refineMatches(
                	warpedKp, refinedMatches, g_refinedHomography
		);

		g_homography = g_roughHomography * g_refinedHomography;
		perspectiveTransform(g_pattern.p2d, g_points2d, g_homography);
	}

	return homographyFound;
}


void parseArguments(int argc, char** argv) {
	const String keys =
		"{ v video  | | path to the video file }"
		"{ r roi    | | select ROI }"
		"{ f fovy   | | vertical field of view in radians }"
		"{ a aspect | | camera aspect ratio }"
	;

	CommandLineParser parser = CommandLineParser(argc, argv, keys);
	parser.about("hello");

	if (parser.has("video"))
		g_videoPath = parser.get<std::string>("video");

	if (parser.has("roi"))
		g_selectRoi = true;

	if (parser.has("fovy"))
		g_fovY = parser.get<float>("fovy");

	if (parser.has("aspect"))
		g_aspect = parser.get<float>("aspect");
}


int main(int argc, char** argv)
{
	parseArguments(argc, argv);

	VideoCapture cap(g_videoPath);

	Mat img;
	cap >> img;

	namedWindow(g_windowName, CV_WINDOW_OPENGL | CV_WINDOW_AUTOSIZE);
	resizeWindow(g_windowName, img.cols, img.rows);

	initPattern(img);

	Mat_<float> cameraMatrix = getCameraMatrix(
		g_fovY, g_aspect, img.cols, img.rows
	);
	g_projectionMat = getProjectionMatrix(cameraMatrix);

	initGL();
	initBgTexture(img);

	setOpenGlDrawCallback(g_windowName, drawGL, (void *)&img);

	while (true)
	{
		cap >> img;

		if (findPattern(img)) {
			solvePnP(g_pattern.p3d, g_points2d, cameraMatrix,
					Mat(), g_raux, g_taux, !g_raux.empty());
		}

		updateWindow(g_windowName);

		if (waitKey(1) == 27) break;
	}

	return 0;
}
