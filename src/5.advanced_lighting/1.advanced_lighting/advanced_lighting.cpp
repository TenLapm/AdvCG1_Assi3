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

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
unsigned int loadTexture(const char* path);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
bool blinn = false;
bool blinnKeyPressed = false;

// camera
Camera camera(glm::vec3(0.0f, 2.0f, 6.0f));
float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// --- XPBD SOFT BODY PHYSICS DATA ---

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

std::vector<Particle> particles;
std::vector<EdgeConstraint> edges;

const float gravity = -9.8f;
const float floorLevel = -0.5f;

// --- COMPLIANCE SETTINGS ---
const float distanceCompliance = 0.001f; // Stiff
const float volumeCompliance = 0.0f;    

float restVolume = 0.0f; // Stores the initial volume of the shape

void initSoftBody() {
    particles.clear();
    edges.clear();

    glm::vec3 offset(0.0f, 4.0f, 0.0f);
    particles.push_back({ glm::vec3(0.0f, -0.5f,  0.0f) + offset, glm::vec3(0.0f), glm::vec3(0.0f), 1.0f }); // 0: Apex (Bottom)
    particles.push_back({ glm::vec3(0.5f,  0.5f,  0.5f) + offset, glm::vec3(0.0f), glm::vec3(0.0f), 1.0f }); // 1: Base Right
    particles.push_back({ glm::vec3(-0.5f, 0.5f,  0.5f) + offset, glm::vec3(0.0f), glm::vec3(0.0f), 1.0f }); // 2: Base Left
    particles.push_back({ glm::vec3(0.0f,  0.5f, -0.5f) + offset, glm::vec3(0.0f), glm::vec3(0.0f), 1.0f }); // 3: Base Back

    int edgePairs[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
    for (int i = 0; i < 6; ++i) {
        float len = glm::distance(particles[edgePairs[i][0]].x, particles[edgePairs[i][1]].x);
        edges.push_back({ edgePairs[i][0], edgePairs[i][1], len });
    }

   
    glm::vec3 p0 = particles[0].x;
    glm::vec3 p1 = particles[1].x;
    glm::vec3 p2 = particles[2].x;
    glm::vec3 p3 = particles[3].x;
    // V = 1/6 * ((x2 - x1) cross (x3 - x1)) dot (x4 - x1)
    restVolume = (1.0f / 6.0f) * glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0);  
}

void updateRenderMesh(float* vertexData) {
    int indices[4][3] = {
        {0, 1, 2}, {0, 3, 1}, {0, 2, 3}, {2, 1, 3}
    };
    float uvs[3][2] = { {0.5f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f} };

    int vIdx = 0;
    for (int f = 0; f < 4; ++f) {
        glm::vec3 p0 = particles[indices[f][0]].x;
        glm::vec3 p1 = particles[indices[f][1]].x;
        glm::vec3 p2 = particles[indices[f][2]].x;
        glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
        glm::vec3 pts[3] = { p0, p1, p2 };
        for (int i = 0; i < 3; ++i) {
            vertexData[vIdx++] = pts[i].x; vertexData[vIdx++] = pts[i].y; vertexData[vIdx++] = pts[i].z;
            vertexData[vIdx++] = normal.x; vertexData[vIdx++] = normal.y; vertexData[vIdx++] = normal.z;
            vertexData[vIdx++] = uvs[i][0]; vertexData[vIdx++] = uvs[i][1];
        }
    }
}

int main()
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
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
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

    initSoftBody();
    float tetRenderData[12 * 8];
    updateRenderMesh(tetRenderData);

    unsigned int tetVAO, tetVBO;
    glGenVertexArrays(1, &tetVAO);
    glGenBuffers(1, &tetVBO);
    glBindVertexArray(tetVAO);
    glBindBuffer(GL_ARRAY_BUFFER, tetVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tetRenderData), tetRenderData, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));

    unsigned int floorTexture = loadTexture(FileSystem::getPath("resources/textures/wood.png").c_str());
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

        // --- 3. XPBD SIMULATION LOOP ---
        int numSubsteps = 10;
        float dt_s = deltaTime / numSubsteps;

        for (int step = 0; step < numSubsteps; ++step)
        {
            // Predict Positions
            for (auto& p : particles) {
                if (p.w > 0.0f) {
                    p.v += glm::vec3(0, gravity, 0) * dt_s;
                    p.p = p.x;
                    p.x += p.v * dt_s;
                }
            }

            // Distance Constraints
            float alpha_dist = distanceCompliance / (dt_s * dt_s);
            for (auto& edge : edges) {
                Particle& p1 = particles[edge.p1];
                Particle& p2 = particles[edge.p2];
                glm::vec3 dir = p1.x - p2.x;
                float len = glm::length(dir);
                if (len < 0.0001f) continue;

                glm::vec3 n = dir / len;
                float C = len - edge.restLength;
                float wSum = p1.w + p2.w;
                float lambda = -C / (wSum + alpha_dist);

                p1.x += lambda * p1.w * n;
                p2.x -= lambda * p2.w * n;
            }

            float alpha_vol = volumeCompliance / (dt_s * dt_s);

            glm::vec3 x0 = particles[0].x;
            glm::vec3 x1 = particles[1].x;
            glm::vec3 x2 = particles[2].x;
            glm::vec3 x3 = particles[3].x;

            float currentVolume = (1.0f / 6.0f) * glm::dot(glm::cross(x1 - x0, x2 - x0), x3 - x0);
            float C_vol = 6.0f * (currentVolume - restVolume);

            // Gradients
            glm::vec3 grad0 = glm::cross(x3 - x1, x2 - x1);
            glm::vec3 grad1 = glm::cross(x2 - x0, x3 - x0);
            glm::vec3 grad2 = glm::cross(x3 - x0, x1 - x0);
            glm::vec3 grad3 = glm::cross(x1 - x0, x2 - x0);

            float wSumVol = particles[0].w * glm::dot(grad0, grad0) +
                particles[1].w * glm::dot(grad1, grad1) +
                particles[2].w * glm::dot(grad2, grad2) +
                particles[3].w * glm::dot(grad3, grad3);

            if (wSumVol > 0.0001f) {
                float lambdaVol = -C_vol / (wSumVol + alpha_vol);

                // Apply volume corrections
                particles[0].x += lambdaVol * particles[0].w * grad0;
                particles[1].x += lambdaVol * particles[1].w * grad1;
                particles[2].x += lambdaVol * particles[2].w * grad2;
                particles[3].x += lambdaVol * particles[3].w * grad3;
            }

            // Ground Collision
            for (auto& p : particles) {
                if (p.x.y < floorLevel) {
                    p.x.y = floorLevel;
                    glm::vec3 lateralVel = p.x - p.p;
                    lateralVel.y = 0;
                    p.p += lateralVel * 0.1f;
                }
            }

            // Update Velocities
            for (auto& p : particles) {
                if (p.w > 0.0f) {
                    p.v = (p.x - p.p) / dt_s;
                }
            }
        }

        updateRenderMesh(tetRenderData);
        glBindBuffer(GL_ARRAY_BUFFER, tetVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(tetRenderData), tetRenderData);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        shader.setMat4("projection", projection);
        shader.setMat4("view", view);
        shader.setVec3("viewPos", camera.Position);
        shader.setVec3("lightPos", lightPos);
        shader.setInt("blinn", blinn);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, floorTexture);

        shader.setMat4("model", glm::mat4(1.0f));
        glBindVertexArray(planeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        shader.setMat4("model", glm::mat4(1.0f));
        glBindVertexArray(tetVAO);
        glDrawArrays(GL_TRIANGLES, 0, 12);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &planeVAO); glDeleteBuffers(1, &planeVBO);
    glDeleteVertexArrays(1, &tetVAO); glDeleteBuffers(1, &tetVBO);
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // --- RESET SOFT BODY ---
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        initSoftBody(); // Rebuild the particles at the starting height
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
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

// utility function for loading a 2D texture from file
// ---------------------------------------------------
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
