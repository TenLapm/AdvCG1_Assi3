#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
unsigned int loadTexture(const char* path);
void renderSphere();
void renderQuad();
void renderFloorPlane();

const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

 
Camera camera(glm::vec3(0.0f, 0.0f, 15.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

std::vector<glm::vec3> kdopAxes = {
    glm::vec3(1, 0, 0),
    glm::vec3(0, 1, 0),
    glm::vec3(0, 0, 1),
    glm::normalize(glm::vec3(1, 1, 1)),
    glm::normalize(glm::vec3(1, -1, 1)),
    glm::normalize(glm::vec3(1, 1, -1)),
    glm::normalize(glm::vec3(1, -1, -1)),
    glm::normalize(glm::vec3(1, 1, 0)),
    glm::normalize(glm::vec3(1, -1, 0)),
    glm::normalize(glm::vec3(1, 0, 1)),
    glm::normalize(glm::vec3(1, 0, -1)),
    glm::normalize(glm::vec3(0, 1, 1)),
    glm::normalize(glm::vec3(0, 1, -1))
};

struct Object {
    glm::vec3 pos;
    glm::vec3 vel;
    float radius;
    float worldMin[13];
    float worldMax[13];
    glm::vec3 color;
};

glm::mat4 getRotation(glm::vec3 N) {
    glm::vec3 up = glm::vec3(0, 1, 0);
    if (fabs(N.y) > 0.999f) up = glm::vec3(1, 0, 0);
    glm::vec3 right = glm::normalize(glm::cross(up, N));
    up = glm::normalize(glm::cross(N, right));

    glm::mat4 rot(1.0f);
    rot[0] = glm::vec4(right, 0.0f);
    rot[1] = glm::vec4(up, 0.0f);
    rot[2] = glm::vec4(N, 0.0f);
    return rot;
}

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL - 26-DOP Physics", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glDisable(GL_CULL_FACE);

    Shader shader("1.1.pbr.vs", "1.1.pbr.fs");
    Shader planeShader("plane.vs", "plane.fs");

    shader.use();
    shader.setFloat("ao", 1.0f);
    shader.setFloat("metallic", 0.1f);
    shader.setFloat("roughness", 0.4f);

    glm::vec3 lightPositions[] = {
        glm::vec3(-10.0f,  10.0f, 10.0f),
        glm::vec3(10.0f,  10.0f, 10.0f),
        glm::vec3(-10.0f, -10.0f, 10.0f),
        glm::vec3(10.0f, -10.0f, 10.0f),
    };
    glm::vec3 lightColors[] = {
        glm::vec3(300.0f, 300.0f, 300.0f),
        glm::vec3(300.0f, 300.0f, 300.0f),
        glm::vec3(300.0f, 300.0f, 300.0f),
        glm::vec3(300.0f, 300.0f, 300.0f)
    };

    srand((unsigned int)time(NULL));
    std::vector<Object> objects;
    for (int i = 0; i < 15; ++i) {
        Object obj;
        obj.pos = glm::vec3((rand() % 100) / 10.0f - 5.0f, (rand() % 100) / 10.0f - 5.0f, (rand() % 100) / 10.0f - 5.0f);
        obj.vel = glm::vec3((rand() % 100) / 20.0f - 2.5f, (rand() % 100) / 20.0f - 2.5f, (rand() % 100) / 20.0f - 2.5f);
        obj.radius = 0.6f + (rand() % 100) / 200.0f;  
        obj.color = glm::vec3((rand() % 100) / 100.0f, (rand() % 100) / 100.0f, (rand() % 100) / 100.0f);
        objects.push_back(obj);
    }

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        float roomSize = 6.0f;

        for (auto& obj : objects) {
            obj.pos += obj.vel * deltaTime;

            for (int i = 0; i < 3; ++i) {
                if (obj.pos[i] > roomSize - obj.radius) {
                    obj.pos[i] = roomSize - obj.radius; obj.vel[i] *= -1.0f;
                }
                if (obj.pos[i] < -roomSize + obj.radius) {
                    obj.pos[i] = -roomSize + obj.radius; obj.vel[i] *= -1.0f;
                }
            }

            for (int i = 0; i < 13; ++i) { 
                float proj = glm::dot(obj.pos, kdopAxes[i]);
                obj.worldMin[i] = proj - obj.radius;
                obj.worldMax[i] = proj + obj.radius;
            }
        }

        for (size_t i = 0; i < objects.size(); ++i) {
            for (size_t j = i + 1; j < objects.size(); ++j) {
                auto& A = objects[i];
                auto& B = objects[j];

                bool intersect = true;
                float minOverlap = 9999.0f;
                glm::vec3 mtv_A(0.0f);

                for (int k = 0; k < 13; ++k) { 
                    if (A.worldMax[k] < B.worldMin[k] || B.worldMax[k] < A.worldMin[k]) {
                        intersect = false;
                        break; 
                    }

                    float overlap1 = A.worldMax[k] - B.worldMin[k]; 
                    float overlap2 = B.worldMax[k] - A.worldMin[k]; 

                    if (overlap1 < minOverlap) {
                        minOverlap = overlap1;
                        mtv_A = -kdopAxes[k]; 
                    }
                    if (overlap2 < minOverlap) {
                        minOverlap = overlap2;
                        mtv_A = kdopAxes[k]; 
                    }
                }

                if (intersect) {
                    A.pos += mtv_A * (minOverlap * 0.5f);
                    B.pos -= mtv_A * (minOverlap * 0.5f);

                    glm::vec3 N = glm::normalize(mtv_A);
                    glm::vec3 vrel = A.vel - B.vel;
                    float vn = glm::dot(vrel, N);

                    if (vn < 0) { 
                        float restitution = 1.0f;
                        float j_imp = -(1.0f + restitution) * vn / 2.0f;
                        A.vel += N * j_imp;
                        B.vel -= N * j_imp;
                    }
                }
            }
        }

        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();

        glDepthMask(GL_TRUE);
        shader.use();
        shader.setMat4("projection", projection);
        shader.setMat4("view", view);
        shader.setVec3("camPos", camera.Position);

        for (unsigned int i = 0; i < 4; ++i) {
            shader.setVec3("lightPositions[" + std::to_string(i) + "]", lightPositions[i]);
            shader.setVec3("lightColors[" + std::to_string(i) + "]", lightColors[i]);
        }

        shader.setVec3("albedo", glm::vec3(0.15f, 0.15f, 0.15f));
        glm::mat4 modelFloor = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -roomSize, 0.0f));
        shader.setMat4("model", modelFloor);
        shader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(modelFloor))));
        renderFloorPlane();

        shader.setVec3("albedo", glm::vec3(0.8f, 0.4f, 0.1f)); 
        for (int x = -1; x <= 1; x += 2) {
            for (int z = -1; z <= 1; z += 2) {
                glm::mat4 modelPillar = glm::translate(glm::mat4(1.0f), glm::vec3(x * roomSize, 0.0f, z * roomSize));
                modelPillar = glm::scale(modelPillar, glm::vec3(0.2f, roomSize, 0.2f));
                shader.setMat4("model", modelPillar);
                shader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(modelPillar))));
                renderSphere();
            }
        }

        for (auto& obj : objects) {
            shader.setVec3("albedo", obj.color);
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, obj.pos);
            model = glm::scale(model, glm::vec3(obj.radius * 0.98f));
            shader.setMat4("model", model);
            shader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));
            renderSphere();
        }

        glDepthMask(GL_FALSE);
        planeShader.use();
        planeShader.setMat4("projection", projection);
        planeShader.setMat4("view", view);

        for (auto& obj : objects) {
            planeShader.setVec4("color", glm::vec4(obj.color + glm::vec3(0.2f), 0.2f)); 

            for (int k = 0; k < 13; ++k) { 
                glm::vec3 N = kdopAxes[k];
                glm::mat4 rot = getRotation(N);

                float scale = obj.radius * 1.15f;

                glm::mat4 modelMax = glm::translate(glm::mat4(1.0f), obj.pos + N * obj.radius);
                modelMax = modelMax * rot * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
                planeShader.setMat4("model", modelMax);
                renderQuad();

                glm::mat4 modelMin = glm::translate(glm::mat4(1.0f), obj.pos - N * obj.radius);
                modelMin = modelMin * rot * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
                planeShader.setMat4("model", modelMin);
                renderQuad();
            }
        }
        glDepthMask(GL_TRUE); 

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos; lastY = ypos;
    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

unsigned int quadVAO = 0;
void renderQuad() {
    if (quadVAO == 0) {
        float quadVertices[] = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
        };
        unsigned int quadVBO;
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

unsigned int sphereVAO = 0;
unsigned int indexCount;
void renderSphere()
{
    if (sphereVAO == 0)
    {
        glGenVertexArrays(1, &sphereVAO);
        unsigned int vbo, ebo;
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);

        std::vector<glm::vec3> positions;
        std::vector<glm::vec2> uv;
        std::vector<glm::vec3> normals;
        std::vector<unsigned int> indices;

        const unsigned int X_SEGMENTS = 64;
        const unsigned int Y_SEGMENTS = 64;
        const float PI = 3.14159265359f;
        for (unsigned int x = 0; x <= X_SEGMENTS; ++x)
        {
            for (unsigned int y = 0; y <= Y_SEGMENTS; ++y)
            {
                float xSegment = (float)x / (float)X_SEGMENTS;
                float ySegment = (float)y / (float)Y_SEGMENTS;
                float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
                float yPos = std::cos(ySegment * PI);
                float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);

                positions.push_back(glm::vec3(xPos, yPos, zPos));
                uv.push_back(glm::vec2(xSegment, ySegment));
                normals.push_back(glm::vec3(xPos, yPos, zPos));
            }
        }

        bool oddRow = false;
        for (unsigned int y = 0; y < Y_SEGMENTS; ++y)
        {
            if (!oddRow)
            {
                for (unsigned int x = 0; x <= X_SEGMENTS; ++x)
                {
                    indices.push_back(y * (X_SEGMENTS + 1) + x);
                    indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
                }
            }
            else
            {
                for (int x = X_SEGMENTS; x >= 0; --x)
                {
                    indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
                    indices.push_back(y * (X_SEGMENTS + 1) + x);
                }
            }
            oddRow = !oddRow;
        }
        indexCount = static_cast<unsigned int>(indices.size());

        std::vector<float> data;
        for (unsigned int i = 0; i < positions.size(); ++i)
        {
            data.push_back(positions[i].x);
            data.push_back(positions[i].y);
            data.push_back(positions[i].z);
            if (normals.size() > 0)
            {
                data.push_back(normals[i].x);
                data.push_back(normals[i].y);
                data.push_back(normals[i].z);
            }
            if (uv.size() > 0)
            {
                data.push_back(uv[i].x);
                data.push_back(uv[i].y);
            }
        }
        glBindVertexArray(sphereVAO);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
        unsigned int stride = (3 + 2 + 3) * sizeof(float);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    }

    glBindVertexArray(sphereVAO);
    glDrawElements(GL_TRIANGLE_STRIP, indexCount, GL_UNSIGNED_INT, 0);
    
}
unsigned int floorVAO = 0;
void renderFloorPlane() {
    if (floorVAO == 0) {
        float planeVertices[] = {
             25.0f, 0.0f,  25.0f,  0.0f, 1.0f, 0.0f,  25.0f,  0.0f,
            -25.0f, 0.0f,  25.0f,  0.0f, 1.0f, 0.0f,   0.0f,  0.0f,
            -25.0f, 0.0f, -25.0f,  0.0f, 1.0f, 0.0f,   0.0f, 25.0f,

             25.0f, 0.0f,  25.0f,  0.0f, 1.0f, 0.0f,  25.0f,  0.0f,
            -25.0f, 0.0f, -25.0f,  0.0f, 1.0f, 0.0f,   0.0f, 25.0f,
             25.0f, 0.0f, -25.0f,  0.0f, 1.0f, 0.0f,  25.0f, 25.0f
        };
        unsigned int floorVBO;
        glGenVertexArrays(1, &floorVAO);
        glGenBuffers(1, &floorVBO);
        glBindVertexArray(floorVAO);
        glBindBuffer(GL_ARRAY_BUFFER, floorVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    }
    glBindVertexArray(floorVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}