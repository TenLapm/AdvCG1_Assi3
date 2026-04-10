#define NOMINMAX
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// --- Global State for Interaction ---
float simWidth, simHeight;
float obsX = 0.4f;
float obsY = 0.5f;
float obsR = 0.15f;
float prevObsX = 0.4f;
float prevObsY = 0.5f;
bool isDragging = false;

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D fluidTexture;
uniform vec2 obsCenter; 
uniform float obsRadius; 
uniform float aspectRatio;

void main() {
    vec4 fluidCol = texture(fluidTexture, TexCoord);

    // Correct UV aspect ratio for a perfect circle
    vec2 uv = TexCoord;
    uv.x *= aspectRatio;
    vec2 center = obsCenter;
    center.x *= aspectRatio;

    float dist = distance(uv, center);

    // Draw the Obstacle on top of the fluid
    if (dist < obsRadius) {
        FragColor = vec4(0.8, 0.8, 0.8, 1.0); // Light Grey Fill
    } else if (dist < obsRadius + 0.005) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black Outline
    } else {
        FragColor = fluidCol; // Fluid Background
    }
}
)";
enum FieldType { U_FIELD = 0, V_FIELD = 1, S_FIELD = 2 };

class Fluid {
public:
    float density;
    int numX, numY, numCells;
    float h;
    std::vector<float> u, v, newU, newV, p, s, m, newM;

    Fluid(float density, int numX, int numY, float h) : density(density), numX(numX + 2), numY(numY + 2), h(h) {
        numCells = this->numX * this->numY;
        u.resize(numCells, 0.0f);
        v.resize(numCells, 0.0f);
        newU.resize(numCells, 0.0f);
        newV.resize(numCells, 0.0f);
        p.resize(numCells, 0.0f);
        s.resize(numCells, 0.0f);
        m.resize(numCells, 1.0f);
        newM.resize(numCells, 0.0f);
    }

    void simulate(float dt, float gravity, int numIters, float overRelaxation) {
        integrate(dt, gravity);
        std::fill(p.begin(), p.end(), 0.0f);
        solveIncompressibility(numIters, dt, overRelaxation);
        extrapolate();
        advectVel(dt);
        advectSmoke(dt);
    }

private:
    void integrate(float dt, float gravity) {
        int n = numY;
        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[i * n + j] != 0.0f && s[i * n + j - 1] != 0.0f)
                    v[i * n + j] += gravity * dt;
            }
        }
    }

    void solveIncompressibility(int numIters, float dt, float overRelaxation) {
        int n = numY;
        float cp = density * h / dt;

        for (int iter = 0; iter < numIters; iter++) {
            for (int i = 1; i < numX - 1; i++) {
                for (int j = 1; j < numY - 1; j++) {
                    if (s[i * n + j] == 0.0f) continue;

                    float sx0 = s[(i - 1) * n + j];
                    float sx1 = s[(i + 1) * n + j];
                    float sy0 = s[i * n + j - 1];
                    float sy1 = s[i * n + j + 1];
                    float sumS = sx0 + sx1 + sy0 + sy1;
                    if (sumS == 0.0f) continue;

                    float div = u[(i + 1) * n + j] - u[i * n + j] +
                        v[i * n + j + 1] - v[i * n + j];

                    float pressure = -div / sumS;
                    pressure *= overRelaxation;
                    p[i * n + j] += cp * pressure;

                    u[i * n + j] -= sx0 * pressure;
                    u[(i + 1) * n + j] += sx1 * pressure;
                    v[i * n + j] -= sy0 * pressure;
                    v[i * n + j + 1] += sy1 * pressure;
                }
            }
        }
    }

    void extrapolate() {
        int n = numY;
        for (int i = 0; i < numX; i++) {
            u[i * n + 0] = u[i * n + 1];
            u[i * n + numY - 1] = u[i * n + numY - 2];
        }
        for (int j = 0; j < numY; j++) {
            v[0 * n + j] = v[1 * n + j];
            v[(numX - 1) * n + j] = v[(numX - 2) * n + j];
        }
    }

    float sampleField(float x, float y, FieldType field) {
        int n = numY;
        float h1 = 1.0f / h;
        float h2 = 0.5f * h;

        x = std::max(std::min(x, numX * h), h);
        y = std::max(std::min(y, numY * h), h);

        float dx = 0.0f; float dy = 0.0f;
        const std::vector<float>* f = nullptr;

        switch (field) {
        case U_FIELD: f = &u; dy = h2; break;
        case V_FIELD: f = &v; dx = h2; break;
        case S_FIELD: f = &m; dx = h2; dy = h2; break;
        }

        int x0 = std::min((int)std::floor((x - dx) * h1), numX - 1);
        float tx = ((x - dx) - x0 * h) * h1;
        int x1 = std::min(x0 + 1, numX - 1);

        int y0 = std::min((int)std::floor((y - dy) * h1), numY - 1);
        float ty = ((y - dy) - y0 * h) * h1;
        int y1 = std::min(y0 + 1, numY - 1);

        float sx = 1.0f - tx;
        float sy = 1.0f - ty;

        return sx * sy * (*f)[x0 * n + y0] +
            tx * sy * (*f)[x1 * n + y0] +
            tx * ty * (*f)[x1 * n + y1] +
            sx * ty * (*f)[x0 * n + y1];
    }

    float avgU(int i, int j) {
        int n = numY;
        return (u[i * n + j - 1] + u[i * n + j] + u[(i + 1) * n + j - 1] + u[(i + 1) * n + j]) * 0.25f;
    }

    float avgV(int i, int j) {
        int n = numY;
        return (v[(i - 1) * n + j] + v[i * n + j] + v[(i - 1) * n + j + 1] + v[i * n + j + 1]) * 0.25f;
    }

    void advectVel(float dt) {
        newU = u; newV = v;
        int n = numY;
        float h2 = 0.5f * h;

        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY; j++) {
                if (s[i * n + j] != 0.0f && s[(i - 1) * n + j] != 0.0f && j < numY - 1) {
                    float x = i * h;
                    float y = j * h + h2;
                    float cu = u[i * n + j];
                    float cv = avgV(i, j);
                    x = x - dt * cu; y = y - dt * cv;
                    newU[i * n + j] = sampleField(x, y, U_FIELD);
                }
                if (s[i * n + j] != 0.0f && s[i * n + j - 1] != 0.0f && i < numX - 1) {
                    float x = i * h + h2;
                    float y = j * h;
                    float cu = avgU(i, j);
                    float cv = v[i * n + j];
                    x = x - dt * cu; y = y - dt * cv;
                    newV[i * n + j] = sampleField(x, y, V_FIELD);
                }
            }
        }
        u = newU; v = newV;
    }

    void advectSmoke(float dt) {
        newM = m;
        int n = numY;
        float h2 = 0.5f * h;

        for (int i = 1; i < numX - 1; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[i * n + j] != 0.0f) {
                    float cu = (u[i * n + j] + u[(i + 1) * n + j]) * 0.5f;
                    float cv = (v[i * n + j] + v[i * n + j + 1]) * 0.5f;
                    float x = i * h + h2 - dt * cu;
                    float y = j * h + h2 - dt * cv;
                    newM[i * n + j] = sampleField(x, y, S_FIELD);
                }
            }
        }
        m = newM;
    }
};

void setupWindTunnel(Fluid& f) {
    int n = f.numY;
    float inVel = 2.0f;
    for (int i = 0; i < f.numX; i++) {
        for (int j = 0; j < f.numY; j++) {
            float solid = 1.0f;
            if (i == 0 || j == 0 || j == f.numY - 1) solid = 0.0f;
            f.s[i * n + j] = solid;
            if (i == 1) f.u[i * n + j] = inVel;
        }
    }
    int pipeH = (int)(0.1f * f.numY);
    int minJ = (int)std::floor(0.5f * f.numY - 0.5f * pipeH);
    int maxJ = (int)std::floor(0.5f * f.numY + 0.5f * pipeH);
    for (int j = minJ; j < maxJ; j++) f.m[1 * n + j] = 0.0f;
}

void enforceWindTunnel(Fluid& f) {
    int n = f.numY;
    float inVel = 1.0f; 

    for (int j = 0; j < f.numY; j++) {
        if (f.s[1 * n + j] != 0.0f) { 
            f.u[1 * n + j] = inVel;
        }
    }

    int pipeH = (int)(0.1f * f.numY); 
    int minJ = (int)std::floor(0.5f * f.numY - 0.5f * pipeH);
    int maxJ = (int)std::floor(0.5f * f.numY + 0.5f * pipeH);
    for (int j = minJ; j < maxJ; j++) {
        f.m[1 * n + j] = 0.0f; 
    }
}

// Updated to compute velocity based on dragging
void setObstacle(Fluid& f, float x, float y, float r, float dt, float prevX, float prevY) {
    float vx = (x - prevX) / dt;
    float vy = (y - prevY) / dt;

    int n = f.numY;
    for (int i = 1; i < f.numX - 2; i++) {
        for (int j = 1; j < f.numY - 2; j++) {
            f.s[i * n + j] = 1.0f; 
            float dx = (i + 0.5f) * f.h - x;
            float dy = (j + 0.5f) * f.h - y;
            if (dx * dx + dy * dy < r * r) {
                f.s[i * n + j] = 0.0f;
                f.m[i * n + j] = 1.0f;
                f.u[i * n + j] = vx;
                f.u[(i + 1) * n + j] = vx;
                f.v[i * n + j] = vy;
                f.v[i * n + j + 1] = vy;
            }
        }
    }
}

unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int id = glCreateShader(type);
    glShaderSource(id, 1, &source, NULL);
    glCompileShader(id);
    return id;
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Eulerian Fluid Sim", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    unsigned int program = glCreateProgram();
    unsigned int vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    float vertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Initialize Simulation Data
    simHeight = 1.0f;
    simWidth = (float)SCR_WIDTH / SCR_HEIGHT * simHeight;
    int res = 100;
    float h = simHeight / res;
    int numX = std::floor(simWidth / h);
    int numY = std::floor(simHeight / h);
    float density = 1000.0f;

    Fluid fluid(density, numX, numY, h);
    setupWindTunnel(fluid);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    std::vector<unsigned char> pixelData(fluid.numX * fluid.numY * 3);
    float dt = 1.0f / 60.0f;

    // Get Uniform Locations
    int obsCenterLoc = glGetUniformLocation(program, "obsCenter");
    int obsRadiusLoc = glGetUniformLocation(program, "obsRadius");
    int aspectRatioLoc = glGetUniformLocation(program, "aspectRatio");

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        setObstacle(fluid, obsX, obsY, obsR, dt, prevObsX, prevObsY);
        prevObsX = obsX;
        prevObsY = obsY;

        enforceWindTunnel(fluid);

        fluid.simulate(dt, 0.0f, 40, 1.9f);

        for (int i = 0; i < fluid.numX; i++) {
            for (int j = 0; j < fluid.numY; j++) {
                int index = (j * fluid.numX + i) * 3;
                float smokeVal = fluid.m[i * fluid.numY + j];

                unsigned char col = (unsigned char)(smokeVal * 255.0f);
                pixelData[index] = col;
                pixelData[index + 1] = col;
                pixelData[index + 2] = col;
            }
        }

        // 4. Upload Data to Texture
        glBindTexture(GL_TEXTURE_2D, texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fluid.numX, fluid.numY, 0, GL_RGB, GL_UNSIGNED_BYTE, pixelData.data());

        // 5. Render Quad
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);

        // Pass Uniforms for the smooth vector circle
        float aspectRatio = (float)SCR_WIDTH / (float)SCR_HEIGHT;
        glUniform2f(obsCenterLoc, obsX / simWidth, obsY / simHeight);
        glUniform1f(obsRadiusLoc, obsR / simHeight);
        glUniform1f(aspectRatioLoc, aspectRatio);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// --- Mouse Input Callbacks ---
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) isDragging = true;
        else if (action == GLFW_RELEASE) isDragging = false;
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (isDragging) {
        // Map window pixel coordinates to our simulation dimensions
        obsX = (float)(xpos / SCR_WIDTH) * simWidth;
        obsY = (float)(1.0 - (ypos / SCR_HEIGHT)) * simHeight; // Y is inverted in GLFW
    }
}