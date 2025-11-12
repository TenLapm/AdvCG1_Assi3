#define GLM_ENABLE_EXPERIMENTAL
#define MINIAUDIO_IMPLEMENTATION
#define STB_EASY_FONT_IMPLEMENTATION
#include "miniAudioMaster/miniaudio.h"
#include "stb_easy_font.h"
#include <glad/glad.h>
#include <stb_image.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include <learnopengl/camera.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/model.h>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>


// Simple camera (kept/used for 3D cube)
const unsigned int INIT_SCR_WIDTH = 1920;
const unsigned int INIT_SCR_HEIGHT = 1080;
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

// globals for window/framebuffer size and cursor
int screenWidth = (int)INIT_SCR_WIDTH;
int screenHeight = (int)INIT_SCR_HEIGHT;
double cursorX = INIT_SCR_WIDTH / 2.0;
double cursorY = INIT_SCR_HEIGHT / 2.0;

ma_engine audioEngine;
ma_sound bgm;
size_t nextBeatIndex = 0;
ma_decoder decoder;     
float audioMeanEnergy = 0.0f;

//edge check
bool gameStarted = false;          
bool prevSpace = false;          
bool prevVolUp = false;            
bool prevVolDown = false;        
float lastEnergyCheck = 0.0f;
float prevEnergy = 0.0f;

float musicVolume = 0.05f;

float range = 0.2f;   // range of time to find peak 0.25 = 25ms
float tipSkip = 0.15f;   // skip this much time after a peak is found
int   maxTargetSpawnPerRange = 1;    
float minHeight = 0.015f;  
int   smoothKernelSamples = 512;     // smoothing size for the abs-envelope
float nextAnalysisAt = 0.0f;


std::vector<float> streamMono;      // holds mono samples
ma_uint64 streamSR = 0;             // decoder cache
ma_uint64 streamLastFrame = 0;      // next unread frame in decoder
size_t maxStreamSamples = 0;        // ring buffer

//debug
double lastAnalysisLogTime = -10.0;

int comboCount = 0;            // current consecutive hits
int bestCombo = 0;             // best (highest) combo reached this session
bool hitThisFrame = false;     // helper to avoid double-counting in a frame
bool missedThisFrame = false;  // helper to avoid multiple resets in a frame

GLuint textVAO = 0, textVBO = 0;
GLuint textProg = 0;
GLint text_uScreenLoc = -1;
GLint text_uColorLoc = -1;

float HP_MAX = 20.0f;        // total health
float hp = HP_MAX;           // current health
float hpBarMargin = 0.03f;   // edge gap
float hpBarHeight = 0.04f;   // bar thickness
GLuint hpVAO = 0, hpVBO = 0;

static void smoothAbsEnvelope(const std::vector<float>& mono,
    int kernel, std::vector<float>& env)
{
    if (mono.empty()) { env.clear(); return; }
    kernel = std::max(1, kernel | 1); 
    int r = kernel / 2;
    env.assign(mono.size(), 0.0f);

    std::vector<double> ps(mono.size() + 1, 0.0);
    for (size_t i = 0; i < mono.size(); ++i) ps[i + 1] = ps[i] + std::fabs(mono[i]);

    for (size_t i = 0; i < mono.size(); ++i) {
        int a = std::max<int>(0, (int)i - r);
        int b = std::min<int>((int)mono.size() - 1, (int)i + r);
        double sum = ps[b + 1] - ps[a];
        env[i] = (float)(sum / double(b - a + 1));
    }
}

static float localMax(const std::vector<float>& env, int idx)
{
    const int N = (int)env.size();
    if (N == 0) return 0.0f;
    float peak = env[idx];

    // walk left to a local minimum “valley”
    float leftMin = peak;
    for (int i = idx - 1; i >= 0; --i) {
        leftMin = std::min(leftMin, env[i]);
        if (env[i] > env[i + 1]) break; // rising again
    }

    // walk right to a local minimum “valley”
    float rightMin = peak;
    for (int i = idx + 1; i < N; ++i) {
        rightMin = std::min(rightMin, env[i]);
        if (env[i] > env[i - 1]) break; // rising again
    }

    float base = std::max(leftMin, rightMin);
    return peak - base;
}

// Find peaks sorted 
static std::vector<int> findPeaks(const std::vector<float>& env,
    float minProm, int wantCount)
{
    std::vector<std::pair<float, int>> cand; // (score, idx)
    const int N = (int)env.size();
    if (N < 3) return {};

    for (int i = 1; i < N - 1; ++i) {
        if (env[i] > env[i - 1] && env[i] >= env[i + 1]) {
            float prom = localMax(env, i);
            if (prom >= minProm) {
                // score: weigh prominence first, then peak height a bit
                float score = prom * 0.8f + env[i] * 0.2f;
                cand.emplace_back(score, i);
            }
        }
    }
    // if not enough, allow minor peaks (lower minProm) to fill
    if ((int)cand.size() < wantCount) {
        for (int i = 1; i < N - 1; ++i) {
            if (env[i] > env[i - 1] && env[i] >= env[i + 1]) {
                float prom = localMax(env, i);
                if (prom > 0.0f) { // any bump
                    float score = prom * 0.6f + env[i] * 0.4f;
                    cand.emplace_back(score, i);
                }
            }
        }
    }

    std::sort(cand.begin(), cand.end(),
        [](auto& a, auto& b) { return a.first > b.first; });

    // unique indices, capped
    std::vector<int> out;
    out.reserve(std::min((int)cand.size(), wantCount));
    for (auto& pr : cand) {
        int idx = pr.second;
        if (std::find(out.begin(), out.end(), idx) == out.end()) {
            out.push_back(idx);
            if ((int)out.size() >= wantCount) break;
        }
    }
    return out;
}


void framebuffer_size_callback(GLFWwindow* window, int w, int h) {
    screenWidth = w;
    screenHeight = h > 0 ? h : 1;
    glViewport(0, 0, screenWidth, screenHeight);
}

void cursor_pos_callback(GLFWwindow* /*window*/, double xpos, double ypos) {
    cursorX = xpos;
    cursorY = ypos;
}

// tiny shader helpers (kept from your code)
static GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetShaderInfoLog(sh, sizeof(buf), nullptr, buf);
        std::cerr << "Shader compile error: " << buf << '\n';
    }
    return sh;
}
static GLuint linkProgram(GLuint vert, GLuint frag) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    GLint ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        std::cerr << "Program link error: " << buf << '\n';
    }
    return prog;
}

struct Target {
    glm::vec3 pos;
    float size;
    float speed;
    bool alive;
};

struct WingMarker {
    glm::vec3 pos;
    float speed;
    bool alive;
    bool left; 
};

float computeAudioEnergy(ma_decoder* decoder, float timeStart, float windowSeconds) {
    ma_uint64 sampleRate = decoder->outputSampleRate;
    ma_uint64 startFrame = static_cast<ma_uint64>(timeStart * sampleRate);
    ma_uint64 frameCount = static_cast<ma_uint64>(windowSeconds * sampleRate);

    std::vector<float> buffer(frameCount * decoder->outputChannels);

    // Seek to the correct starting point
    if (ma_decoder_seek_to_pcm_frame(decoder, startFrame) != MA_SUCCESS)
        return 0.0f;

    // Read frames with correct signature
    ma_uint64 framesRead = 0;
    if (ma_decoder_read_pcm_frames(decoder, buffer.data(), frameCount, &framesRead) != MA_SUCCESS)
        return 0.0f;

    if (framesRead == 0) return 0.0f;

    double energySum = 0.0;
    size_t samples = static_cast<size_t>(framesRead * decoder->outputChannels);

    for (size_t i = 0; i < samples; ++i)
        energySum += fabs(buffer[i]);

    return static_cast<float>(energySum / samples);
}
void DrawText(const char* text, float x, float y, float r, float g, float b) {
    if (!textProg) return;

    char stbBuffer[99999];
    int num_quads = stb_easy_font_print((float)x, (float)y, (char*)text, NULL, stbBuffer, sizeof(stbBuffer));
    if (num_quads <= 0) return;

    float* v = (float*)stbBuffer;
    std::vector<float> triVerts;
    triVerts.reserve(num_quads * 6 * 2);
    for (int q = 0; q < num_quads; ++q) {
        float x0 = v[q * 8 + 0], y0 = v[q * 8 + 1];
        float x1 = v[q * 8 + 2], y1 = v[q * 8 + 3];
        float x2 = v[q * 8 + 4], y2 = v[q * 8 + 5];
        float x3 = v[q * 8 + 6], y3 = v[q * 8 + 7];

        // tri 1
        triVerts.push_back(x0); triVerts.push_back(y0);
        triVerts.push_back(x1); triVerts.push_back(y1);
        triVerts.push_back(x2); triVerts.push_back(y2);
        // tri 2
        triVerts.push_back(x2); triVerts.push_back(y2);
        triVerts.push_back(x3); triVerts.push_back(y3);
        triVerts.push_back(x0); triVerts.push_back(y0);
    }

    // Upload verts
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glBufferData(GL_ARRAY_BUFFER, triVerts.size() * sizeof(float), triVerts.data(), GL_DYNAMIC_DRAW);

    // Bind VAO + attribute (robust on all drivers)
    glBindVertexArray(textVAO);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // State for 2D UI
    GLboolean depthWas = glIsEnabled(GL_DEPTH_TEST);
    GLboolean cullWas = glIsEnabled(GL_CULL_FACE);

    if (depthWas) glDisable(GL_DEPTH_TEST);
    if (cullWas)  glDisable(GL_CULL_FACE);

    GLint polygonMode[2]; glGetIntegerv(GL_POLYGON_MODE, polygonMode);
    if (polygonMode[0] != GL_FILL || polygonMode[1] != GL_FILL)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw
    glUseProgram(textProg);
    glUniform2f(text_uScreenLoc, (float)screenWidth, (float)screenHeight);
    glUniform3f(text_uColorLoc, r, g, b);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(num_quads * 6));

    // Restore
    glUseProgram(0);
    glDisable(GL_BLEND);
    if (polygonMode[0] != GL_FILL || polygonMode[1] != GL_FILL)
        glPolygonMode(GL_FRONT_AND_BACK, polygonMode[0]); 
    if (cullWas)  glEnable(GL_CULL_FACE);
    if (depthWas) glEnable(GL_DEPTH_TEST);
    glBindVertexArray(0);
}



int main() {
    srand((unsigned)time(nullptr));
    stbi_set_flip_vertically_on_load(true);
    // GLFW init
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "Camera", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        glfwTerminate();
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    std::cout << "Camera initial position: (" << camera.Position.x << ", " << camera.Position.y << ", " << camera.Position.z << ")\n";

    const float halfSize = 0.5f;
    float squareVerts[12] = {
        -halfSize, -halfSize, 0.0f,
         halfSize, -halfSize, 0.0f,
         halfSize,  halfSize, 0.0f,
        -halfSize,  halfSize, 0.0f
    };
    // --- Audio init ---
    if (ma_engine_init(NULL, &audioEngine) != MA_SUCCESS) {
        std::cerr << "Failed to initialize miniaudio engine\n";
        return -1;
    }

    const char* kBGMPath = "../../resources/audio/bgm.mp3";
    const char* kHitPath = "../../resources/audio/hit.mp3";

    if (ma_sound_init_from_file(&audioEngine, kBGMPath, MA_SOUND_FLAG_STREAM, NULL, NULL, &bgm) != MA_SUCCESS) {
        std::cerr << "Failed to load bgm.mp3\n";
        return -1;
    }
    ma_sound_set_volume(&bgm, musicVolume);

    ma_decoder_config dcfg = ma_decoder_config_init(ma_format_f32, 0, 0);
    if (ma_decoder_init_file(kBGMPath, &dcfg, &decoder) != MA_SUCCESS) {
        std::cerr << "Failed to init decoder for analysis\n";
        return -1;
    }

    streamSR = decoder.outputSampleRate;
    streamMono.clear();
    streamLastFrame = 0;

    maxStreamSamples = (size_t)std::ceil((double)streamSR * (range * 2.0 + 0.1));
    streamMono.reserve(std::min<size_t>(maxStreamSamples, (size_t)streamSR * 5));


    GLuint squareVAO = 0, squareVBO = 0;
    glGenVertexArrays(1, &squareVAO);
    glGenBuffers(1, &squareVBO);
    glBindVertexArray(squareVAO);
    glBindBuffer(GL_ARRAY_BUFFER, squareVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(squareVerts), squareVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    // HP bar geometry 
    glGenVertexArrays(1, &hpVAO);
    glGenBuffers(1, &hpVBO);
    glBindVertexArray(hpVAO);
    glBindBuffer(GL_ARRAY_BUFFER, hpVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, nullptr, GL_DYNAMIC_DRAW); 
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    float cubeEdgeVerts[] = {
        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f
    };
    unsigned int cubeEdgeIndices[] = {
        0,1, 1,2, 2,3, 3,0,
        4,5, 5,6, 6,7, 7,4,
        0,4, 1,5, 2,6, 3,7
    };
    GLuint cubeEdgeVAO = 0, cubeEdgeVBO = 0, cubeEdgeEBO = 0;
    glGenVertexArrays(1, &cubeEdgeVAO);
    glGenBuffers(1, &cubeEdgeVBO);
    glGenBuffers(1, &cubeEdgeEBO);
    glBindVertexArray(cubeEdgeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeEdgeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeEdgeVerts), cubeEdgeVerts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEdgeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeEdgeIndices), cubeEdgeIndices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    const char* vs_cube = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform mat4 uMVP;
        void main() { gl_Position = uMVP * vec4(aPos,1.0); }
    )";
    const char* fs_cube = R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec3 uColor;
        void main() { FragColor = vec4(uColor,1.0); }
    )";
    GLuint vsc = compileShader(GL_VERTEX_SHADER, vs_cube);
    GLuint fsc = compileShader(GL_FRAGMENT_SHADER, fs_cube);
    GLuint progCube = linkProgram(vsc, fsc);
    glDeleteShader(vsc);
    glDeleteShader(fsc);
    GLint uMVPLoc = glGetUniformLocation(progCube, "uMVP");
    GLint uColorLoc_cube = glGetUniformLocation(progCube, "uColor");

    const char* vs_square = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform float uInvAspect;
        void main() { gl_Position = vec4(aPos.x * uInvAspect, aPos.y, aPos.z, 1.0); }
    )";
    const char* fs_square = R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec3 uColor;
        void main() { FragColor = vec4(uColor,1.0); }
    )";
    GLuint vsq = compileShader(GL_VERTEX_SHADER, vs_square);
    GLuint fsq = compileShader(GL_FRAGMENT_SHADER, fs_square);
    GLuint progSquare = linkProgram(vsq, fsq);
    glDeleteShader(vsq);
    glDeleteShader(fsq);
    GLint uInvAspectLoc_square = glGetUniformLocation(progSquare, "uInvAspect");
    GLint uColorLoc_square = glGetUniformLocation(progSquare, "uColor");

    const char* vs_circle = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        uniform vec2 uCenter;
        uniform float uRadius;
        uniform float uInvAspect;
        void main(){ vec2 p = aPos * uRadius + uCenter; gl_Position = vec4(p.x * uInvAspect, p.y, 0.0, 1.0); }
    )";
    const char* fs_circle = R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec3 uColor;
        void main() { FragColor = vec4(uColor,1.0); }
    )";
    GLuint vsc2 = compileShader(GL_VERTEX_SHADER, vs_circle);
    GLuint fsc2 = compileShader(GL_FRAGMENT_SHADER, fs_circle);
    GLuint progCircle = linkProgram(vsc2, fsc2);
    glDeleteShader(vsc2);
    glDeleteShader(fsc2);
    GLint uCenterLoc = glGetUniformLocation(progCircle, "uCenter");
    GLint uRadiusLoc = glGetUniformLocation(progCircle, "uRadius");
    GLint uInvAspectLoc_circle = glGetUniformLocation(progCircle, "uInvAspect");
    GLint uColorLoc_circle = glGetUniformLocation(progCircle, "uColor");

    glLineWidth(3.0f);

    Shader cursorShader("1.model_loading.vs", "1.model_loading.fs");
    Model cursorModel("../../resources/objects/Game/prometheus.obj");
    std::cout << "Loaded model: " << cursorModel.meshes.size() << " meshes\n";
    for (auto& m : cursorModel.meshes) {
        std::cout << "Mesh has " << m.vertices.size() << " vertices\n";
        glm::vec3 minv(FLT_MAX), maxv(-FLT_MAX);
        for (auto& v : m.vertices) {
            minv = glm::min(minv, v.Position);
            maxv = glm::max(maxv, v.Position);
        }
        glm::vec3 size = maxv - minv;
        std::cout << "Bounds: min " << glm::to_string(minv)
            << " max " << glm::to_string(maxv)
            << " size " << glm::to_string(size) << "\n";
    }
    glUseProgram(cursorShader.ID);
    std::cout << "model=" << glGetUniformLocation(cursorShader.ID, "model")
        << " view=" << glGetUniformLocation(cursorShader.ID, "view")
        << " proj=" << glGetUniformLocation(cursorShader.ID, "projection") << "\n";

    const char* vs_text = R"(
    #version 330 core
    layout(location = 0) in vec2 aPos;   
    uniform vec2 uScreen;              
    void main() {
        vec2 ndc;
        ndc.x = (aPos.x / uScreen.x) * 2.0 - 1.0;
        ndc.y = 1.0 - (aPos.y / uScreen.y) * 2.0;
        gl_Position = vec4(ndc.xy, 0.0, 1.0);
    }
    )";

    const char* fs_text = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 uColor;
    void main() {
        FragColor = vec4(uColor, 1.0);
    }
    )";

    GLuint v = compileShader(GL_VERTEX_SHADER, vs_text);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs_text);
    textProg = linkProgram(v, f);
    glDeleteShader(v);
    glDeleteShader(f);
    text_uScreenLoc = glGetUniformLocation(textProg, "uScreen");
    text_uColorLoc = glGetUniformLocation(textProg, "uColor");

    glGenVertexArrays(1, &textVAO);
    glGenBuffers(1, &textVBO);
    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);



    const float cursorScale = 0.4f;
    const float cursorModelDistance = 6.0f;
    float modelScaleNormalize = 0.4f;

    const float collisionZRange = 1.2f;
    const float respawnZ = -30.0f;
    const float nearTestZ = 0.0f;
    const float defaultCubeSize = 0.25f;
    const float defaultCubeSpeed = 16.0f;
    const float worldSpread = 4.0f;

    std::vector<Target> targets;

    std::vector<WingMarker> markers;
    const int numPairs = 2;
    const float spacing = 12.0f;

    for (int pair = 0; pair < numPairs; ++pair) {
        float zOffset = -20.0f - pair * spacing;
        for (int side = 0; side < 2; ++side) {
            WingMarker w;
            w.pos = glm::vec3((side == 0 ? -3.0f : 3.0f), 0.0f, zOffset);
            w.speed = defaultCubeSpeed;
            w.alive = true;
            w.left = (side == 0);
            markers.push_back(w);
        }
    }

    auto spawnTarget = [&](glm::vec3& outPos,
        const glm::mat4& projection,
        const glm::mat4& view,
        float cubeSizeLocal,
        float halfSizePreAspect,
        float invAspect) -> bool
        {
            const int   MAX_TRIES = 256;
            const float radius_pixels = 12.0f;

            const float radius_ndc_y = (2.0f * radius_pixels) / float(screenHeight);

            float radius_pre = (invAspect != 0.0f) ? (radius_ndc_y / invAspect) : radius_ndc_y;

            float margin = std::max(0.001f, radius_pre);
            float minPre = -halfSizePreAspect + margin;
            float maxPre = halfSizePreAspect - margin;

            for (int t = 0; t < MAX_TRIES; ++t) {
                float cx = ((rand() / (float)RAND_MAX) * 2.0f - 1.0f) * worldSpread;
                float cy = ((rand() / (float)RAND_MAX) * 2.0f - 1.0f) * worldSpread;

                glm::vec3 testPos(cx, cy, nearTestZ);

                glm::vec4 clip = projection * view * glm::vec4(testPos, 1.0f);
                if (clip.w <= 0.0f) continue;

                glm::vec3 ndc = glm::vec3(clip) / clip.w;
                float preX = (invAspect != 0.0f) ? (ndc.x / invAspect) : ndc.x;
                float preY = ndc.y;

                glm::vec3 offX = testPos + glm::vec3(cubeSizeLocal * 0.5f, 0.0f, 0.0f);
                glm::vec4 clipOff = projection * view * glm::vec4(offX, 1.0f);

                float projRadiusPre = 0.01f;
                if (clipOff.w > 0.0f) {
                    glm::vec3 ndcOff = glm::vec3(clipOff) / clipOff.w;
                    float preXOff = (invAspect != 0.0f) ? (ndcOff.x / invAspect) : ndcOff.x;
                    projRadiusPre = std::max(0.01f, fabs(preXOff - preX));
                }

                if (preX - projRadiusPre >= minPre && preX + projRadiusPre <= maxPre &&
                    preY - projRadiusPre >= minPre && preY + projRadiusPre <= maxPre)
                {
                    outPos = glm::vec3(cx, cy, respawnZ); 
                    return true;
                }
            }

            // fallback
            outPos = glm::vec3(0.0f, 0.0f, respawnZ);
            return false;
        };



    double lastTime = glfwGetTime();

    // main loop
    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = float(now - lastTime);
        lastTime = now;

        hitThisFrame = false;
        missedThisFrame = false;

        // --- Music timing check ---
        float musicTime = 0.0f;
        ma_sound_get_cursor_in_seconds(&bgm, &musicTime);

        int spaceNow = glfwGetKey(window, GLFW_KEY_SPACE);
        bool spacePressed = (spaceNow == GLFW_PRESS) && !prevSpace;
        prevSpace = (spaceNow == GLFW_PRESS);

        int volUpNow = glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET);   // ]
        int volDownNow = glfwGetKey(window, GLFW_KEY_LEFT_BRACKET);  // [
        bool volUpPressed = (volUpNow == GLFW_PRESS) && !prevVolUp;
        bool volDownPressed = (volDownNow == GLFW_PRESS) && !prevVolDown;
        prevVolUp = (volUpNow == GLFW_PRESS);
        prevVolDown = (volDownNow == GLFW_PRESS);

        if (!gameStarted && spacePressed) {
            nextBeatIndex = 0;
            nextAnalysisAt = 0.0f;
            ma_sound_seek_to_pcm_frame(&bgm, 0);
            ma_sound_start(&bgm);

            // Reset decoder & analysis timing
            ma_decoder_seek_to_pcm_frame(&decoder, 0);
            lastEnergyCheck = 0.0f;
            prevEnergy = 0.0f;
            audioMeanEnergy = 0.0f;

            gameStarted = true;
            std::cout << "Game started! Music playing.\n";
        }


        // Volume edge controls
        if (volUpPressed) {
            musicVolume = glm::clamp(musicVolume + 0.05f, 0.0f, 1.0f);
            ma_sound_set_volume(&bgm, musicVolume);
            std::cout << "Volume: " << musicVolume << "\n";
        }
        if (volDownPressed) {
            musicVolume = glm::clamp(musicVolume - 0.05f, 0.0f, 1.0f);
            ma_sound_set_volume(&bgm, musicVolume);
            std::cout << "Volume: " << musicVolume << "\n";
        }
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            nextAnalysisAt = 0.0f;
            targets.clear();
            nextBeatIndex = 0;

            ma_sound_stop(&bgm);
            ma_sound_seek_to_pcm_frame(&bgm, 0);

            ma_decoder_seek_to_pcm_frame(&decoder, 0);
            lastEnergyCheck = 0.0f;
            prevEnergy = 0.0f;
            audioMeanEnergy = 0.0f;

            gameStarted = false;
			hp = HP_MAX;
            std::cout << "Reset. Press SPACE to start.\n";
        }


        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
            screenWidth > 0 ? float(screenWidth) / float(screenHeight) : 1.0f,
            0.5f, 200.0f);
        glm::mat4 view = camera.GetViewMatrix();
        float invAspect = (screenWidth > 0) ? (float(screenHeight) / float(screenWidth)) : 1.0f;

        if (musicTime >= nextAnalysisAt) {
            if (streamSR == 0) streamSR = decoder.outputSampleRate;
            ma_uint64 currentFrame = (ma_uint64)(musicTime * (double)streamSR);

            if (currentFrame < streamLastFrame) {
                // decode backwards position
                if (ma_decoder_seek_to_pcm_frame(&decoder, currentFrame) == MA_SUCCESS) {
                    streamMono.clear();
                    streamLastFrame = currentFrame;
                }
                else {
                    nextAnalysisAt = musicTime + tipSkip;
                    continue;
                }
            }

            ma_uint64 framesToRead = currentFrame > streamLastFrame ? (currentFrame - streamLastFrame) : 0;
            if (framesToRead > 0) {
                // allocate interleaved buffer once
                size_t interSamples = (size_t)framesToRead * decoder.outputChannels;
                std::vector<float> inter;
                try { inter.resize(interSamples); }
                catch (...) { inter.clear(); framesToRead = 0; }

                ma_uint64 framesRead = 0;
                if (framesToRead > 0 &&
                    ma_decoder_read_pcm_frames(&decoder, inter.data(), framesToRead, &framesRead) == MA_SUCCESS &&
                    framesRead > 0) {

                    // convert framesRead to mono and append to streamMono
                    streamMono.reserve(std::min(streamMono.size() + (size_t)framesRead, maxStreamSamples));
                    for (ma_uint64 f = 0; f < framesRead; ++f) {
                        double acc = 0.0;
                        for (ma_uint32 c = 0; c < decoder.outputChannels; ++c) {
                            acc += inter[(size_t)f * decoder.outputChannels + c];
                        }
                        streamMono.push_back((float)(acc / (double)decoder.outputChannels));
                    }
                    streamLastFrame += framesRead;
                }
            }

            // trim buffer
            if (streamMono.size() > maxStreamSamples) {
                size_t removeCount = streamMono.size() - maxStreamSamples;
                streamMono.erase(streamMono.begin(), streamMono.begin() + removeCount);
            }

            // only analyze when we actually have a full window
            size_t windowSamples = (size_t)std::max<ma_uint64>(1, (ma_uint64)std::round(tipSkip * (double)streamSR));
            if (streamMono.size() >= windowSamples) {
                // copy last windowSamples into envMono 
                std::vector<float> envMono;
                envMono.assign(streamMono.end() - windowSamples, streamMono.end());

                // smoothing kernel size scaled to sr 
                const float smoothingMs = 12.0f;
                int kernel = std::max(3, (int)std::round((smoothingMs * 0.001f) * (float)streamSR));
                if ((kernel & 1) == 0) --kernel;
                kernel = std::min(kernel, std::max(3, (int)envMono.size() / 16));

                std::vector<float> env;
                smoothAbsEnvelope(envMono, kernel, env);
                if (!env.empty()) {
                    // normalize
                    float envMax = *std::max_element(env.begin(), env.end());
                    if (envMax > 0.0f) {
                        for (float& v : env) v /= envMax;
                    }

                    // normalized mean
                    float envMean = 0.0f;
                    for (float v : env) envMean += v;
                    envMean /= (float)env.size();
                    double var = 0.0;
                    for (float v : env) { double d = v - envMean; var += d * d; }
                    float envStd = (env.size() > 1) ? (float)std::sqrt(var / (env.size() - 1)) : 0.0f;
                    float kStd = 0.6f;
                    float adaptiveThresh = glm::clamp(envMean + kStd * envStd, 0.03f, 0.9f);

                    int wantPeaks = maxTargetSpawnPerRange;
                    auto peakIdx = findPeaks(env, adaptiveThresh, wantPeaks);

                    // throttle logs
                    double nowT = glfwGetTime();
                    if (nowT - lastAnalysisLogTime >= 1.0) {
                        std::cout << "ANALYSIS: now=" << musicTime << " samples=" << env.size()
                            << " mean=" << envMean << " std=" << envStd << " thr=" << adaptiveThresh
                            << " peaks=" << peakIdx.size() << "\n";
                        lastAnalysisLogTime = nowT;
                    }

                    // cap total number of targets
                    const size_t MAX_TOTAL_TARGETS = 128;
                    for (int idx : peakIdx) {
                        if (targets.size() >= MAX_TOTAL_TARGETS) break;

                        glm::vec3 p;
                        if (spawnTarget(p, projection, view, defaultCubeSize, halfSize, (float)screenHeight / (float)screenWidth)) {
                            targets.push_back(Target{ p, defaultCubeSize, defaultCubeSpeed, true });
                        }
                    }
                }
            }

            nextAnalysisAt = musicTime + range;
        } 



        for (auto& m : markers) {
            m.pos.z += m.speed * dt;
            if (m.pos.z > 2.5f) {
                m.pos.z = -30.0f;
            }
        }

        // update targets
        for (auto& t : targets) {
            if (!t.alive) continue;
            t.pos.z += t.speed * dt;
            if (t.pos.z > 2.5f) {
                t.alive = false;
                if (!missedThisFrame) {
                    comboCount = 0;
                    missedThisFrame = true;
                    std::cout << "Miss! combo reset\n";
                }
            }
        }

        // clear
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        // draw targets
        for (auto& t : targets) {
            if (!t.alive) continue;
            glEnable(GL_DEPTH_TEST);
            glUseProgram(progCube);
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, t.pos);
            model = glm::scale(model, glm::vec3(t.size));
            glm::mat4 mvp = projection * view * model;
            glUniformMatrix4fv(uMVPLoc, 1, GL_FALSE, glm::value_ptr(mvp));
            glUniform3f(uColorLoc_cube, 0.85f, 0.2f, 0.15f);
            glBindVertexArray(cubeEdgeVAO);
            glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
            glUseProgram(0);

        }

        for (auto& m : markers) {
            glUseProgram(progCube);
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, m.pos);
            model = glm::scale(model, glm::vec3(0.5f, 3.0f, 0.5f));

            if (m.left)
                model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0, 1, 0));
            else
                model = glm::rotate(model, glm::radians(-45.0f), glm::vec3(0, 1, 0));

            glm::mat4 mvp = projection * view * model;
            glUniformMatrix4fv(uMVPLoc, 1, GL_FALSE, glm::value_ptr(mvp));
            glUniform3f(uColorLoc_cube, 0.2f, 0.9f, 0.9f);
            glBindVertexArray(cubeEdgeVAO);
            glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
            glUseProgram(0);
        }

        GLboolean depthWasEnabled = glIsEnabled(GL_DEPTH_TEST);
        if (depthWasEnabled) glDisable(GL_DEPTH_TEST);

        float left = -halfSize + hpBarMargin;
        float rightF = halfSize - hpBarMargin;        
        float bottom = -halfSize + hpBarMargin;       
        float top = bottom + hpBarHeight;

        float t = (HP_MAX > 0.0f) ? (hp / HP_MAX) : 0.0f;
        t = glm::clamp(t, 0.0f, 1.0f);
        float right = left + (rightF - left) * t;

        auto drawQuadPre = [&](float L, float B, float R, float T, float r, float g, float b)
            {
                float verts[6 * 3] = {
                    L, B, 0.0f,  R, B, 0.0f,  R, T, 0.0f,
                    R, T, 0.0f,  L, T, 0.0f,  L, B, 0.0f
                };
                glUseProgram(progSquare);
                glUniform1f(uInvAspectLoc_square, (screenHeight > 0) ? (float)screenHeight / (float)screenWidth : 1.0f);
                glUniform3f(uColorLoc_square, r, g, b);

                glBindVertexArray(hpVAO);
                glBindBuffer(GL_ARRAY_BUFFER, hpVBO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                glBindVertexArray(0);
                glUseProgram(0);
                if (depthWasEnabled) glEnable(GL_DEPTH_TEST);
            };

        drawQuadPre(left, bottom, rightF, top, 0.12f, 0.12f, 0.12f);

        float rCol, gCol, bCol = 0.15f;
        if (t > 0.5f) {               
            float k = (t - 0.5f) / 0.5f;
            rCol = k; gCol = 1.0f;
        }
        else {                     
            float k = t / 0.5f;
            rCol = 1.0f; gCol = k;
        }

        if (right > left) {
            drawQuadPre(left, bottom, right, top, rCol, gCol, bCol);
        }


        // draw square outline
        glUseProgram(progSquare);
        glUniform1f(uInvAspectLoc_square, invAspect);
        glUniform3f(uColorLoc_square, 0.95f, 0.35f, 0.9f);
        glBindVertexArray(squareVAO);
        glDrawArrays(GL_LINE_LOOP, 0, 4);
        glBindVertexArray(0);
        glUseProgram(0);

        float ndcX = float((cursorX / double(screenWidth)) * 2.0 - 1.0);
        float ndcY = float(1.0 - (cursorY / double(screenHeight)) * 2.0);
        float center_pre_x = (invAspect != 0.0f) ? (ndcX / invAspect) : ndcX;
        float center_pre_y = ndcY;

        const float radius_pixels = 12.0f;
        float final_radius_ndc = (2.0f * radius_pixels) / float(screenHeight);
        float radius_pre = (invAspect != 0.0f) ? (final_radius_ndc / invAspect) : final_radius_ndc;
        if (radius_pre > halfSize - 0.001f) radius_pre = halfSize - 0.001f;

        float minX = -halfSize + radius_pre;
        float maxX = halfSize - radius_pre;
        float minY = -halfSize + radius_pre;
        float maxY = halfSize - radius_pre;
        if (center_pre_x < minX) center_pre_x = minX;
        if (center_pre_x > maxX) center_pre_x = maxX;
        if (center_pre_y < minY) center_pre_y = minY;
        if (center_pre_y > maxY) center_pre_y = maxY;

        // collision
        for (auto& t : targets) {
            if (!t.alive) continue;
            if (!(t.pos.z >= -collisionZRange && t.pos.z <= collisionZRange)) continue;

            glm::vec4 clip = projection * view * glm::vec4(t.pos, 1.0f);
            if (clip.w <= 0.0f) continue;

            glm::vec3 ndc = glm::vec3(clip) / clip.w;
            float cube_pre_x = (invAspect != 0.0f) ? (ndc.x / invAspect) : ndc.x;
            float cube_pre_y = ndc.y;

            glm::vec3 offsetWorld = t.pos + glm::vec3(t.size * 0.5f, 0.0f, 0.0f);
            glm::vec4 clipOff = projection * view * glm::vec4(offsetWorld, 1.0f);
            float projRadius_pre = 0.01f;
            if (clipOff.w > 0.0f) {
                glm::vec3 ndcOff = glm::vec3(clipOff) / clipOff.w;
                float off_pre_x = (invAspect != 0.0f) ? (ndcOff.x / invAspect) : ndcOff.x;
                float rx = fabs(off_pre_x - cube_pre_x);
                projRadius_pre = (rx <= 0.00001f) ? 0.01f : rx;
            }

            float dx = center_pre_x - cube_pre_x;
            float dy = center_pre_y - cube_pre_y;
            float dist = sqrtf(dx * dx + dy * dy);

            if (dist <= (projRadius_pre + radius_pre)) {
                // a real hit
                if (!hitThisFrame) {
                    comboCount += 1;
                    if (comboCount > bestCombo) bestCombo = comboCount;
                    std::cout << "Hit! combo=" << comboCount << " best=" << bestCombo << "\n";
                    hitThisFrame = true;
                }

                ma_engine_play_sound(&audioEngine, kHitPath, NULL);

                t.alive = false;
            }
        }


        float cursor_ndc_x = center_pre_x * invAspect;
        float cursor_ndc_y = center_pre_y;

        glm::mat4 invPV = glm::inverse(projection * view);

        glm::vec4 ndcNear(cursor_ndc_x, cursor_ndc_y, -1.0f, 1.0f);
        glm::vec4 ndcFar(cursor_ndc_x, cursor_ndc_y, 1.0f, 1.0f);

        glm::vec4 wNear = invPV * ndcNear;
        wNear /= wNear.w;
        glm::vec4 wFar = invPV * ndcFar;
        wFar /= wFar.w;

        glm::vec3 worldNear = glm::vec3(wNear);
        glm::vec3 worldFar = glm::vec3(wFar);

        glm::vec3 rayDir = glm::normalize(worldFar - worldNear);
        glm::vec3 cursorWorldPos = camera.Position + rayDir * cursorModelDistance;


        float totalScale = cursorScale * modelScaleNormalize;

        glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(totalScale));

        glm::mat3 viewRot = glm::mat3(view);
        glm::mat3 invRot = glm::transpose(viewRot);
        glm::mat4 R = glm::mat4(invRot);

        glm::mat4 T = glm::translate(glm::mat4(1.0f), cursorWorldPos);
        glm::mat4 modelCursor = T * R * S;

        //glEnable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        cursorShader.use();
        cursorShader.setMat4("projection", projection);
        cursorShader.setMat4("view", view);
        cursorShader.setMat4("model", modelCursor);

        cursorModel.Draw(cursorShader);


        // remove dead targets
        targets.erase(std::remove_if(targets.begin(), targets.end(),
            [](const Target& t) { return !t.alive; }), targets.end());

        if (!gameStarted) {
            DrawText("Press SPACE to start", 8.0f, 12.0f, 1.0f, 1.0f, 1.0f);
        }
        
        std::string s = "Combo: " + std::to_string(comboCount) + "  Best: " + std::to_string(bestCombo);
        DrawText(s.c_str(), 8.0f, 36.0f, 1.0f, 0.9f, 0.2f);

        DrawText("TEST", screenWidth * 0.5f - 40.0f, screenHeight * 0.5f, 1, 1, 1);

        if (missedThisFrame) {
            hp = std::max(0.0f, hp - 1.0f);
        }
        if (hitThisFrame) {
            hp = std::min(HP_MAX, hp + 0.25f);
        }
        if (hp <= 0) {
            nextAnalysisAt = 0.0f;
            targets.clear();
            nextBeatIndex = 0;

            ma_sound_stop(&bgm);
            ma_sound_seek_to_pcm_frame(&bgm, 0);

            ma_decoder_seek_to_pcm_frame(&decoder, 0);
            lastEnergyCheck = 0.0f;
            prevEnergy = 0.0f;
            audioMeanEnergy = 0.0f;

            gameStarted = false;
            std::cout << "Reset. Press SPACE to start.\n";
        }

        // swap/poll
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteProgram(progCube);
    glDeleteProgram(progSquare);
    glDeleteProgram(progCircle);

    glDeleteProgram(cursorShader.ID);

    for (auto& mesh : cursorModel.meshes) {
        glDeleteVertexArrays(1, &mesh.VAO);
    }
    for (auto& tex : cursorModel.textures_loaded) {
        glDeleteTextures(1, &tex.id);
    }

    glDeleteBuffers(1, &textVBO);
    glDeleteVertexArrays(1, &textVAO);
    if (textProg) glDeleteProgram(textProg);


    glDeleteVertexArrays(1, &squareVAO);
    glDeleteBuffers(1, &squareVBO);

    glDeleteVertexArrays(1, &cubeEdgeVAO);
    glDeleteBuffers(1, &cubeEdgeVBO);
    glDeleteBuffers(1, &cubeEdgeEBO);
    glDeleteBuffers(1, &hpVBO);
    glDeleteVertexArrays(1, &hpVAO);

    ma_sound_uninit(&bgm);
    ma_decoder_uninit(&decoder);
    ma_engine_uninit(&audioEngine);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
