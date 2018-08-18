#include "common.hpp"


using namespace cv;

Mat_<float> getCameraMatrix(float fovY, float aspect, float w, float h)
{
	float fovX = 2.0 * atan(tan(fovY * 0.5) * aspect);
	float fx = w / tan(fovX * 0.5);
	float fy = h / tan(fovY * 0.5);
	return (Mat_<float>(3, 3) << fx, 0, w * 0.5, 0, fy, h * 0.5, 0, 0, 1);
}


Matx44f getProjectionMatrix(Mat_<float> cam)
{
	float cx = cam.at<float>(0, 2);
	float cy = cam.at<float>(1, 2);
	float fx = cam.at<float>(0, 0);
	float fy = cam.at<float>(1, 1);
	float w = cx * 2.0;
	float h = cy * 2.0;
	float near = 0.01;
	float far  = 100;

	Matx44f p;

	p.val[0] = -2.0 * fx / w;
	p.val[1] = 0;
	p.val[2] = 0;
	p.val[3] = 0;

	p.val[4] = 0;
	p.val[5] = 2.0 * fy / h;
	p.val[6] = 0;
	p.val[7] = 0;

	p.val[8] = 2.0 * cx / w - 1.0;
	p.val[9] = 2.0 * cy / h - 1.0;
	//p.val[8] = 0;
	//p.val[9] = 0;
	p.val[10] = -(far + near) / (far - near);
	p.val[11] = -1;

	p.val[12] = 0;
	p.val[13] = 0;
	p.val[14] = -2.0 * far * near / (far - near);
	p.val[15] = 0;

	return p;
}


GLuint compileShader(GLuint type, const std::string& src)
{
	GLuint shader = glCreateShader(type);
	const char *c_str = src.c_str();
	glShaderSource(shader, 1, &c_str, NULL);
	glCompileShader(shader);
	GLint compiled;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
	if (!compiled) {
		GLint length;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
		std::string errorLog(length, ' ');
		glGetShaderInfoLog(shader, length, &length, &errorLog[0]);
		std::cout << "Shader compilation error\n";
		std::cout << errorLog << "\n";
		return false;
	}
	return shader;
}


std::string g_vertexShaderBgSrc =
R"(
#version 130

in vec4 position;
in vec4 textureCoordsIn;

varying vec2 textureCoords;


void main()
{
	gl_Position = position;
	textureCoords = textureCoordsIn.xy;
}
)";


std::string g_fragmentShaderBgSrc =
R"(
#version 130

varying vec2 textureCoords;
uniform sampler2D textureData;

void main()
{
	gl_FragColor = texture2D(textureData, textureCoords);
}
)";


GLfloat g_vertCoordsBg[] = {
	-1.0, -1.0,
	 1.0, -1.0,
	-1.0,  1.0,
	 1.0,  1.0,
};


GLfloat g_texCoordsBg[] = {
	 0.0,  1.0,
	 1.0,  1.0,
	 0.0,  0.0,
	 1.0,  0.0,
};


GLuint g_textureBg;
GLuint g_vertCoordBufBg;
GLuint g_texCoordBufBg;
GLuint g_programBg;


void initBgGlData()
{
	GLuint vertexShaderBg = compileShader(
		GL_VERTEX_SHADER, g_vertexShaderBgSrc
	);
	GLuint fragmentShaderBg = compileShader(
		GL_FRAGMENT_SHADER, g_fragmentShaderBgSrc
	);
	g_programBg = glCreateProgram();
	glAttachShader(g_programBg, vertexShaderBg);
	glAttachShader(g_programBg, fragmentShaderBg);
	glLinkProgram(g_programBg);

	glGenTextures(1, &g_textureBg);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	glGenBuffers(1, &g_vertCoordBufBg);
	glBindBuffer(GL_ARRAY_BUFFER, g_vertCoordBufBg);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertCoordsBg),
			g_vertCoordsBg, GL_STATIC_DRAW);

	glGenBuffers(1, &g_texCoordBufBg);
	glBindBuffer(GL_ARRAY_BUFFER, g_texCoordBufBg);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_texCoordsBg),
			g_texCoordsBg, GL_STATIC_DRAW);
}


void initBgTexture(Mat& img)
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_textureBg);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0,
			GL_BGR, GL_UNSIGNED_BYTE, img.data);
	const GLchar * texUniformName = "textureData";
	GLint texUniformLoc = glGetUniformLocation(g_programBg, texUniformName);
	glUniform1i(texUniformLoc, 0);

	glBindTexture(GL_TEXTURE_2D, 0);
}


void drawBg(Mat* img)
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_textureBg);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img->cols, img->rows,
			GL_BGR, GL_UNSIGNED_BYTE, img->data);

	glUseProgram(g_programBg);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, g_vertCoordBufBg);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, g_texCoordBufBg);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glBindTexture(GL_TEXTURE_2D, 0);

	//glUseProgram(0);
}
