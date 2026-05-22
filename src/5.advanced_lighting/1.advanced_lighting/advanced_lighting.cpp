#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <tuple>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
unsigned int loadTexture(const char* path);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

int grabbedParticle = -1;
float grabDistance = 0.0f;
bool rightMouseDown = false;

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
bool blinn = false;
bool blinnKeyPressed = false;

Camera camera(glm::vec3(0.0f, 2.0f, 6.0f));
float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

struct Particle {
    glm::vec3 x;
    glm::vec3 p;
    glm::vec3 v;
    float w;
};

struct EdgeConstraint {
    int p1, p2;
    float restLength;
};

struct Tet {
    int idx[4];
    float restVolume;
};

std::vector<Particle> particles;
std::vector<EdgeConstraint> edges;
std::vector<Tet> tets;
std::vector<glm::ivec3> surfaceFaces; 

std::vector<float> renderData;
unsigned int tetVAO, tetVBO;

const float gravity = -9.8f;
const float floorLevel = -0.5f;

const float distanceCompliance = 0.0000f; 
const float volumeCompliance = 0.0000f;

float restVolume = 0.0f; 

void initSoftBody(const std::string& filepath) {
    particles.clear();
    edges.clear();
    tets.clear();
    surfaceFaces.clear();

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cout << "Failed to load tet mesh: " << filepath << std::endl;
        return;
    }

    int numVerts, numTets;
    file >> numVerts >> numTets;

    glm::vec3 offset(0.0f, 4.0f, 0.0f); 
    float scale = 0.05f;

    for (int i = 0; i < numVerts; ++i) {
        glm::vec3 pos;
        file >> pos.x >> pos.y >> pos.z;
        pos *= scale;
        particles.push_back({ pos + offset, glm::vec3(0.0f), glm::vec3(0.0f), 1.0f });
    }

    std::map<std::pair<int, int>, bool> uniqueEdges;
    std::map<std::tuple<int, int, int>, int> faceCounts;

    for (int i = 0; i < numTets; ++i) {
        Tet t;
        file >> t.idx[0] >> t.idx[1] >> t.idx[2] >> t.idx[3];

        glm::vec3 p0 = particles[t.idx[0]].x;
        glm::vec3 p1 = particles[t.idx[1]].x;
        glm::vec3 p2 = particles[t.idx[2]].x;
        glm::vec3 p3 = particles[t.idx[3]].x;

        t.restVolume = (1.0f / 6.0f) * glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0);

        if (t.restVolume < 0.0f) {
            std::swap(t.idx[0], t.idx[1]);
            t.restVolume = -t.restVolume;
        }
        tets.push_back(t);

        int edgePairs[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
        for (int e = 0; e < 6; ++e) {
            int a = std::min(t.idx[edgePairs[e][0]], t.idx[edgePairs[e][1]]);
            int b = std::max(t.idx[edgePairs[e][0]], t.idx[edgePairs[e][1]]);
            uniqueEdges[{a, b}] = true;
        }

        int faceTriangles[4][3] = { {0,1,2}, {0,3,1}, {0,2,3}, {2,1,3} };
        for (int f = 0; f < 4; ++f) {
            int v0 = t.idx[faceTriangles[f][0]];
            int v1 = t.idx[faceTriangles[f][1]];
            int v2 = t.idx[faceTriangles[f][2]];

            int arr[3] = { v0, v1, v2 };
            std::sort(arr, arr + 3);
            faceCounts[{arr[0], arr[1], arr[2]}]++;
        }
    }

    for (auto const& [pair, exists] : uniqueEdges) {
        float len = glm::distance(particles[pair.first].x, particles[pair.second].x);
        edges.push_back({ pair.first, pair.second, len });
    }

    for (int i = 0; i < numTets; ++i) {
        int faceTriangles[4][3] = { {0,1,2}, {0,3,1}, {0,2,3}, {2,1,3} };
        for (int f = 0; f < 4; ++f) {
            int v0 = tets[i].idx[faceTriangles[f][0]];
            int v1 = tets[i].idx[faceTriangles[f][1]];
            int v2 = tets[i].idx[faceTriangles[f][2]];

            int arr[3] = { v0, v1, v2 };
            std::sort(arr, arr + 3);
            if (faceCounts[{arr[0], arr[1], arr[2]}] == 1) {
                surfaceFaces.push_back(glm::ivec3(v0, v1, v2));
            }
        }
    }
    renderData.resize(surfaceFaces.size() * 3 * 8);
}

void updateRenderMesh() {
    glm::vec2 uvs[3] = { {0.5f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f} };
    int idx = 0;

    for (const auto& face : surfaceFaces) {
        glm::vec3 p0 = particles[face.x].x;
        glm::vec3 p1 = particles[face.y].x;
        glm::vec3 p2 = particles[face.z].x;

        glm::vec3 crossProduct = glm::cross(p1 - p0, p2 - p0);
        float crossLen = glm::length(crossProduct);
        glm::vec3 normal = crossLen > 1e-8f ? (crossProduct / crossLen) : glm::vec3(0.0f, 1.0f, 0.0f);

        glm::vec3 pts[3] = { p0, p1, p2 };

        for (int i = 0; i < 3; ++i) {
            renderData[idx++] = pts[i].x; renderData[idx++] = pts[i].y; renderData[idx++] = pts[i].z;
            renderData[idx++] = normal.x; renderData[idx++] = normal.y; renderData[idx++] = normal.z;
            renderData[idx++] = uvs[i].x; renderData[idx++] = uvs[i].y;
        }
    }
}int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL - XPBD Volume Preservation", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetMouseButtonCallback(window, mouse_button_callback);    
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Shader shader("1.advanced_lighting.vs", "1.advanced_lighting.fs");

    float planeVertices[] = {
         10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
        -10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,   0.0f,  0.0f,
        -10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,
         10.0f, -0.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
        -10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,
         10.0f, -0.5f, -10.0f,  0.0f, 1.0f, 0.0f,  10.0f, 10.0f
    };
    unsigned int planeVAO, planeVBO;
    glGenVertexArrays(1, &planeVAO);
    glGenBuffers(1, &planeVBO);
    glBindVertexArray(planeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));

    initSoftBody("D:/Uni Works/Year3/Advcg1_Assi3/AdvCG1_Assi3/resources/objects/horse_softbody/tet_model.txt"); // Adjust path as needed
    updateRenderMesh();

    glGenVertexArrays(1, &tetVAO);
    glGenBuffers(1, &tetVBO);
    glBindVertexArray(tetVAO);
    glBindBuffer(GL_ARRAY_BUFFER, tetVBO);
    glBufferData(GL_ARRAY_BUFFER, renderData.size() * sizeof(float), renderData.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));

    unsigned int floorTexture = loadTexture("D:/Uni Works/Year3/Advcg1_Assi3/AdvCG1_Assi3/resources/textures/wood.png");
    shader.use();
    shader.setInt("texture1", 0);
    glm::vec3 lightPos(0.0f, 5.0f, 0.0f);

    lastFrame = static_cast<float>(glfwGetTime());

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        if (deltaTime > 0.05f) deltaTime = 0.05f;

        processInput(window);
        if (grabbedParticle != -1) {
            glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
            glm::mat4 view = camera.GetViewMatrix();
            float x = (2.0f * lastX) / SCR_WIDTH - 1.0f;
            float y = 1.0f - (2.0f * lastY) / SCR_HEIGHT;
            glm::vec4 ray_clip(x, y, -1.0f, 1.0f);
            glm::vec4 ray_eye = glm::inverse(projection) * ray_clip;
            ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0f, 0.0f);
            glm::vec3 ray_wor = glm::normalize(glm::vec3(glm::inverse(view) * ray_eye));

            glm::vec3 targetPos = camera.Position + ray_wor * grabDistance;
            particles[grabbedParticle].x = targetPos;
            particles[grabbedParticle].p = targetPos; 
            particles[grabbedParticle].v = glm::vec3(0.0f);
        }

        int numSubsteps = 10;
        float dt_s = deltaTime / numSubsteps;

        for (int step = 0; step < numSubsteps; ++step)
        {
            for (auto& p : particles) {
                if (p.w > 0.0f) {
                    p.v += glm::vec3(0, gravity, 0) * dt_s;
                    p.p = p.x;
                    p.x += p.v * dt_s;
                }
            }

            float alpha_dist = distanceCompliance / (dt_s * dt_s);
            for (auto& edge : edges) {
                Particle& p1 = particles[edge.p1];
                Particle& p2 = particles[edge.p2];
                glm::vec3 dir = p1.x - p2.x;
                float len = glm::length(dir);
                if (len < 1e-6f) continue;

                glm::vec3 n = dir / len;
                float C = len - edge.restLength;
                float wSum = p1.w + p2.w;
                float lambda = -C / (wSum + alpha_dist);

                p1.x += lambda * p1.w * n;
                p2.x -= lambda * p2.w * n;
            }

            float alpha_vol = volumeCompliance / (dt_s * dt_s);

            for (auto& t : tets) {
                Particle& p0 = particles[t.idx[0]];
                Particle& p1 = particles[t.idx[1]];
                Particle& p2 = particles[t.idx[2]];
                Particle& p3 = particles[t.idx[3]];

                float currentVolume = (1.0f / 6.0f) * glm::dot(glm::cross(p1.x - p0.x, p2.x - p0.x), p3.x - p0.x);
                float C_vol = 6.0f * (currentVolume - t.restVolume);

                glm::vec3 grad0 = glm::cross(p3.x - p1.x, p2.x - p1.x);
                glm::vec3 grad1 = glm::cross(p2.x - p0.x, p3.x - p0.x);
                glm::vec3 grad2 = glm::cross(p3.x - p0.x, p1.x - p0.x);
                glm::vec3 grad3 = glm::cross(p1.x - p0.x, p2.x - p0.x);

                float wSumVol = p0.w * glm::dot(grad0, grad0) +
                    p1.w * glm::dot(grad1, grad1) +
                    p2.w * glm::dot(grad2, grad2) +
                    p3.w * glm::dot(grad3, grad3);

                if (wSumVol > 1e-15f) {
                    float lambdaVol = -C_vol / (wSumVol + alpha_vol);
                    p0.x += lambdaVol * p0.w * grad0;
                    p1.x += lambdaVol * p1.w * grad1;
                    p2.x += lambdaVol * p2.w * grad2;
                    p3.x += lambdaVol * p3.w * grad3;
                }
            }

            float restitution = 1.0f;
            float friction = 0.1f;
            for (auto& p : particles) {
                if (p.x.y < floorLevel) {
                    p.x.y = floorLevel;

                    p.p.y = p.x.y;

                    glm::vec3 lateralVel = p.x - p.p;
                    p.p.x = p.x.x - (lateralVel.x * 0.9f);
                    p.p.z = p.x.z - (lateralVel.z * 0.9f);
                }
            }

            for (auto& p : particles) {
                if (p.w > 0.0f) {
                    p.v = (p.x - p.p) / dt_s;
                    p.v *= 0.999f;
                }
            }
            
        }

        updateRenderMesh();
        glBindBuffer(GL_ARRAY_BUFFER, tetVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, renderData.size() * sizeof(float), renderData.data());

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        shader.setMat4("projection", projection);
        shader.setMat4("view", view);
        shader.setVec3("viewPos", camera.Position);
        shader.setVec3("lightPos", camera.Position);
        shader.setInt("blinn", blinn);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, floorTexture);

        shader.setMat4("model", glm::mat4(1.0f));
        glBindVertexArray(planeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        shader.setMat4("model", glm::mat4(1.0f));
        glBindVertexArray(tetVAO);
        glDrawArrays(GL_TRIANGLES, 0, surfaceFaces.size() * 3);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &planeVAO); glDeleteBuffers(1, &planeVBO);
    glDeleteVertexArrays(1, &tetVAO); glDeleteBuffers(1, &tetVBO);
    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        initSoftBody("D:/Uni Works/Year3/Advcg1_Assi3/AdvCG1_Assi3/resources/objects/horse_softbody/tet_model.txt");
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.ProcessKeyboard(RIGHT, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS && !blinnKeyPressed)
    {
        blinn = !blinn;
        blinnKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_RELEASE) blinnKeyPressed = false;
}
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    if (rightMouseDown) {
        camera.ProcessMouseMovement(xoffset, yoffset);
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) rightMouseDown = true;
        else if (action == GLFW_RELEASE) rightMouseDown = false;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
            glm::mat4 view = camera.GetViewMatrix();
            float x = (2.0f * lastX) / SCR_WIDTH - 1.0f;
            float y = 1.0f - (2.0f * lastY) / SCR_HEIGHT; 
            glm::vec4 ray_clip(x, y, -1.0f, 1.0f);
            glm::vec4 ray_eye = glm::inverse(projection) * ray_clip;
            ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0f, 0.0f);
            glm::vec3 ray_wor = glm::normalize(glm::vec3(glm::inverse(view) * ray_eye));

            float minRadius = 0.5f; 
            float closestDist = 9999.0f;

            for (int i = 0; i < particles.size(); ++i) {
                glm::vec3 p = particles[i].x;
                glm::vec3 v = p - camera.Position;
                float t = glm::dot(v, ray_wor);
                if (t > 0.0f) {
                    glm::vec3 proj = camera.Position + ray_wor * t;
                    float dist = glm::length(p - proj);
                    if (dist < minRadius && t < closestDist) {
                        closestDist = t;
                        grabbedParticle = i;
                        grabDistance = t;
                    }
                }
            }

            if (grabbedParticle != -1) {
                particles[grabbedParticle].w = 0.0f;
            }
        }
        else if (action == GLFW_RELEASE) {
            if (grabbedParticle != -1) {
                particles[grabbedParticle].w = 1.0f;
                grabbedParticle = -1;
            }
        }
    }
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

unsigned int loadTexture(char const * path)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT); // for this tutorial: use GL_CLAMP_TO_EDGE to prevent semi-transparent borders. Due to interpolation it takes texels from next repeat 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}
