#include "common.hpp"
#include <opencv2/face.hpp>


using namespace cv;
using namespace cv::face;


std::string g_videoPath;
std::string g_cascadePath = "../training/haarcascade_frontalface_alt2.xml";
std::string g_modelPath = "../face_landmark_model.dat";
float g_fovY = 1.3;
float g_aspect = 1.35;


std::string g_vertexShaderMeshSrc =
R"(
#version 130

in vec4 position;
in vec3 vertexColor;
out vec3 fragmentColor;

void main()
{
	gl_Position = gl_ModelViewProjectionMatrix * position;
	fragmentColor = vertexColor;
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


float g_objH = 1.25;
float g_objW = 2.75;
float g_objD = 2.75;


GLfloat g_vertCoordsMesh[] = {
	 0,       g_objH,  g_objD,
	-g_objW,  g_objH,  0,
	-g_objW, -g_objH,  0,
	 0,      -g_objH,  g_objD,
	 0,       g_objH,  g_objD,
	 g_objW,  g_objH,  0,
	 g_objW, -g_objH,  0,
	 0,      -g_objH,  g_objD
};


GLfloat g_vertColsMesh[] = {
	0.2,  0.0,  0.9,
	0.0,  0.0,  0.6,
	0.0,  0.0,  0.6,
	0.2,  0.0,  0.9,

	0.2,  0.0,  0.9,
	0.0,  0.0,  0.6,
	0.0,  0.0,  0.6,
	0.2,  0.0,  0.9,
};


GLuint g_programMesh;
GLuint g_vertCoordBufMesh;
GLuint g_vertColBufMesh;

Mat g_raux, g_taux;
Matx44f g_projectionMat;


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

	glEnable(GL_LINE_SMOOTH);
	glLineWidth(4.0);
	glDrawArrays(GL_LINE_STRIP, 0, 8);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	//glUseProgram(0);
}


Matx44f getModelViewMatrix()
{
	Mat_<float> tvec;
	g_taux.convertTo(tvec, CV_32F);

	Mat rvec(3, 1, CV_32F);
	rvec.at<float>(0) = -g_raux.at<float>(0, 0);
	rvec.at<float>(1) = -g_raux.at<float>(1, 0);
	rvec.at<float>(2) =  g_raux.at<float>(2, 0);

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


void drawFaceSerment(int end, int* i, bool isClosed, std::vector<Point>& pt,
			const Scalar& col, const Mat& img,
			const std::vector<Point2f>& points)
{
	pt.clear();
	for(; *i < end; (*i)++) pt.push_back(points.at(*i));
	polylines(img, pt, isClosed, col, 2);
}


void drawFace(std::vector<Point>& dpt, const Mat& img,
		const std::vector<Point2f>& pts)
{
	int j = 0;
	Scalar col(50, 0, 255);
	dpt.clear();
	Scalar col2(255, 0, 0);
	
	// enclosing face shape
	drawFaceSerment(17, &j, false, dpt, col, img, pts);
	// right eyebrow
	drawFaceSerment(22, &j, false, dpt, col, img, pts);
	// left eyebrow
	drawFaceSerment(27, &j, false, dpt, col, img, pts);
	// nose
	drawFaceSerment(36, &j, false, dpt, col, img, pts);
	Point p0 = pts.at(30);
	Point p1 = pts.at(35);
	line(img, p0, p1, col);
	// right eye
	drawFaceSerment(42, &j, true, dpt, col, img, pts);
	// left eye
	drawFaceSerment(48, &j, true, dpt, col, img, pts);
	// enclosing lips shape
	drawFaceSerment(60, &j, true, dpt, col, img, pts);
	// inner lips shape
	drawFaceSerment(68, &j, true, dpt, col, img, pts);
}


static bool faceDetector(InputArray image, OutputArray faces,
				CascadeClassifier *faceCascade)
{
	Mat gray;

	if (image.channels() > 1)
		cvtColor(image, gray, COLOR_BGR2GRAY);
	else
		gray = image.getMat().clone();

	equalizeHist(gray, gray);

	std::vector<Rect> faces_;
	faceCascade->detectMultiScale(gray, faces_, 1.2, 3,
					CASCADE_SCALE_IMAGE, Size(30, 30));
	Mat(faces_).copyTo(faces);
	return true;
}


std::vector<Point3f> getObjectPoints()
{
	// 27, 28, 29, 30 - nose points downwards
	// 23 - eyebrow point, left in face space
	// 20 - eyebrow point, right in face space

	std::vector<Point3f> points;

	// nose
	//points.push_back(Point3f(0, -0.5, 0.0));
	points.push_back(Point3f(0, -1.0, 0.0));
	points.push_back(Point3f(0, -1.5, 0.0));
	points.push_back(Point3f(0, -2.0, 0.0));
	
	// eyebrows
	points.push_back(Point3f( 1.0, 0, 0.0));
	points.push_back(Point3f(-1.0, 0, 0.0));

	// offset
	for (auto &p : points) {
		p.y -= 1.5;
	}

	return points;
}


void parseArguments(int argc, char** argv) {
	const String keys =
		"{ v video   | | path to the video file }"
		"{ f fovy    | | vertical field of view in radians }"
		"{ a aspect  | | camera aspect ratio }"
		"{ c cascade | | path to the face cascade xml file }"
		"{ m model   | | path to the trained model }"
	;

	CommandLineParser parser = CommandLineParser(argc, argv, keys);
	parser.about("hello");

	g_videoPath = parser.get<std::string>("video");

	if (parser.has("fovy"))
		g_fovY = parser.get<float>("fovy");

	if (parser.has("aspect"))
		g_aspect = parser.get<float>("aspect");

	if (parser.has("cascade"))
		g_cascadePath = parser.get<std::string>("cascade");

	if (parser.has("model"))
		g_modelPath = parser.get<std::string>("model");
}


int main(int argc, char** argv)
{
	parseArguments(argc, argv);

	VideoCapture cap;
	
	if (g_videoPath.empty())
		cap.open(0);
	else
		cap.open(g_videoPath);

	CascadeClassifier faceCascade;
	faceCascade.load(g_cascadePath);

	FacemarkKazemi::Params params;
	Ptr<FacemarkKazemi> facemark = FacemarkKazemi::create(params);
	facemark->setFaceDetector((FN_FaceDetector)faceDetector, &faceCascade);
	facemark->loadModel(g_modelPath);

	std::vector<Rect> faces;
	std::vector<std::vector<Point2f>> facialPoints;
	std::vector<Point3f> objectPoints = getObjectPoints();
	std::vector<Point2f> imagePoints;


	std::string windowName = "Facemarks";
	namedWindow(windowName, CV_WINDOW_OPENGL | CV_WINDOW_AUTOSIZE);

	Mat img;
        cap >> img;
	resizeWindow(windowName, img.cols, img.rows);

	initGL();
	initBgTexture(img);

	Mat_<float> cameraMatrix = getCameraMatrix(
		g_fovY, g_aspect, img.cols, img.rows
	);
	g_projectionMat = getProjectionMatrix(cameraMatrix);
	std::vector<Point> drawPoints;

	setOpenGlDrawCallback(windowName, drawGL, (void *)&img);

	while (true)
	{
		cap >> img;

		faces.clear();
		facialPoints.clear();
		facemark->getFaces(img, faces);

		if (faces.size() > 0) {
			int i = 0;
			rectangle(img,faces[i], Scalar(200, 0, 100), 2);

			if (facemark->fit(img, faces, facialPoints)) {
				drawFace(drawPoints, img, facialPoints[i]);
				
				imagePoints.clear();
				//imagePoints.push_back(facialPoints[i][27]);
				imagePoints.push_back(facialPoints[i][28]);
				imagePoints.push_back(facialPoints[i][29]);
				imagePoints.push_back(facialPoints[i][30]);
				imagePoints.push_back(facialPoints[i][23]);
				imagePoints.push_back(facialPoints[i][20]);

				solvePnP(objectPoints, imagePoints, cameraMatrix,
					Mat(), g_raux, g_taux, !g_raux.empty());
			}
		}
#ifdef DEBUG
		else {
			std::cout << "Failed to detect a face\n";
		}
#endif
		updateWindow(windowName);

		if (waitKey(1) == 27) break;
	}
	return 0;
}
