#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/animator.h>
#include <learnopengl/model_animation.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <stb_image.h>

// --- GLOBALS FOR ANIMATION ACCESS ---
Animation* pIdleAnim = nullptr;
Animation* pWalkAnim = nullptr;
Animation* pRunAnim = nullptr;
Animation* pDeathAnim = nullptr;

// --- ENEMY SETTINGS ---
float ENEMY_SPAWN_RADIUS_MIN = 5.0f;
float ENEMY_SPAWN_RADIUS_MAX = 20.0f;
float ENEMY_SPEED = 3.5f;
float ENEMY_HP = 1.0f;
float ENEMY_HITBOX_RADIUS = 0.6f; 
float ENEMY_BULLET_SPEED = 5.0f;

// --- WAVE SETTINGS ---
int currentWave = 1;
int enemiesSpawnedInWave = 0;
int enemiesToSpawnTotal = 0;

// --- GAME STATE ---
bool gameStarted = false;
bool playerIsDead = false; 
float playerDeathTimer = 0.0f;

// --- MAP SETTINGS ---
struct Obstacle {
    glm::vec3 position;
    glm::vec3 size;
};
std::vector<Obstacle> mapObstacles;

struct Enemy {
    glm::vec3 position;
    float hp;
    float speed;
    float shootTimer;
    float shootInterval;

    Animator animator;
    bool isDead;
    float deadTimeCounter; 

    Enemy() : animator(nullptr), isDead(false), deadTimeCounter(0.0f) {}
};

struct Bullet {
    glm::vec3 position;
    glm::vec3 direction;
    float speed;
    float life;
    bool isPlayerBullet;
};

std::vector<Enemy> enemies;
std::vector<Bullet> bullets;

glm::vec3 currentGunBarrelPos(0.0f);

// --- TIME VARIABLES ---
float timeScale = 0.0f;
float mouseMoveAmt = 0.0f;

// --- PROTOTYPES ---
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
bool processInput(GLFWwindow* window);
void SpawnEnemyInFOV();
void ShootPlayer(glm::vec3 targetPos);
void ShootEnemy(Enemy& enemy, glm::vec3 playerPos, glm::vec3 enemyGunPos);
void CreateMap();
void ManageWaves(float deltaTime);
void ResetGame();
void TriggerPlayerDeath();

// --- MATH HELPERS ---
bool CheckWallCollision(glm::vec3 position, float radius, const Obstacle& obs) {
    glm::vec3 halfSize = obs.size / 2.0f;
    glm::vec3 min = obs.position - halfSize;
    glm::vec3 max = obs.position + halfSize;
    glm::vec3 closest = glm::vec3(
        std::max(min.x, std::min(position.x, max.x)),
        std::max(min.y, std::min(position.y, max.y)),
        std::max(min.z, std::min(position.z, max.z))
    );
    float distance = glm::distance(closest, position);
    return distance < radius;
}

bool IsCollidingWithMap(glm::vec3 position, float radius) {
    for (const auto& wall : mapObstacles) {
        if (CheckWallCollision(position, radius, wall)) return true;
    }
    return false;
}

bool IsInsideFOV(glm::vec3 targetPos, const Camera& cam, float fovDeg) {
    glm::vec3 toTarget = glm::normalize(targetPos - cam.Position);
    glm::vec3 camFront = glm::normalize(cam.Front);
    float dot = glm::clamp(glm::dot(camFront, toTarget), -1.0f, 1.0f);
    float angle = glm::degrees(acos(dot));
    return angle < (fovDeg / 1.5f);
}

glm::vec3 MoveWithSliding(glm::vec3 currentPos, glm::vec3 velocity, float radius) {
    glm::vec3 nextPos = currentPos;
    nextPos.x += velocity.x;
    if (IsCollidingWithMap(nextPos, radius)) nextPos.x = currentPos.x;
    nextPos.z += velocity.z;
    if (IsCollidingWithMap(nextPos, radius)) nextPos.z = currentPos.z;
    return nextPos;
}

glm::vec3 GetSmartPath(glm::vec3 startPos, glm::vec3 targetPos) {
    glm::vec3 forward = glm::normalize(targetPos - startPos);
    forward.y = 0.0f;
    float checkDist = 2.5f;
    if (!IsCollidingWithMap(startPos + forward * checkDist, ENEMY_HITBOX_RADIUS)) return forward;
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 tryRight = glm::normalize(forward + right);
    if (!IsCollidingWithMap(startPos + tryRight * checkDist, ENEMY_HITBOX_RADIUS)) return tryRight;
    glm::vec3 tryLeft = glm::normalize(forward - right);
    if (!IsCollidingWithMap(startPos + tryLeft * checkDist, ENEMY_HITBOX_RADIUS)) return tryLeft;
    if (!IsCollidingWithMap(startPos + right * checkDist, ENEMY_HITBOX_RADIUS)) return right;
    if (!IsCollidingWithMap(startPos - right * checkDist, ENEMY_HITBOX_RADIUS)) return -right;
    return forward;
}

float GetRayAABBIntersection(glm::vec3 rayOrigin, glm::vec3 rayDir, const Obstacle& obs) {
    glm::vec3 halfSize = obs.size / 2.0f;
    glm::vec3 boxMin = obs.position - halfSize;
    glm::vec3 boxMax = obs.position + halfSize;
    glm::vec3 dirInv = 1.0f / rayDir;
    float t1 = (boxMin.x - rayOrigin.x) * dirInv.x;
    float t2 = (boxMax.x - rayOrigin.x) * dirInv.x;
    float tMin = std::min(t1, t2);
    float tMax = std::max(t1, t2);
    t1 = (boxMin.y - rayOrigin.y) * dirInv.y;
    t2 = (boxMax.y - rayOrigin.y) * dirInv.y;
    tMin = std::max(tMin, std::min(t1, t2));
    tMax = std::min(tMax, std::max(t1, t2));
    t1 = (boxMin.z - rayOrigin.z) * dirInv.z;
    t2 = (boxMax.z - rayOrigin.z) * dirInv.z;
    tMin = std::max(tMin, std::min(t1, t2));
    tMax = std::min(tMax, std::max(t1, t2));
    if (tMax >= tMin && tMin > 0.0f) return tMin;
    return -1.0f;
}

float GetRaySphereIntersection(glm::vec3 rayOrigin, glm::vec3 rayDir, glm::vec3 sphereCenter, float sphereRadius) {
    glm::vec3 oc = rayOrigin - sphereCenter;
    float b = glm::dot(oc, rayDir);
    float c = glm::dot(oc, oc) - sphereRadius * sphereRadius;
    if (c > 0.0f && b > 0.0f) return -1.0f;
    float disc = b * b - c;
    if (disc < 0.0f) return -1.0f;
    return -b - sqrt(disc);
}

// --- GLOBALS ---
const unsigned int SCR_WIDTH = 1600;
const unsigned int SCR_HEIGHT = 800;
glm::vec3 playerPos(0.0f, -0.4f, 0.0f);
float playerSpeed = 2.5f;
float playerRadius = 0.5f;
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f;
float lastFrame = 0.0f;
float lastShotTime = -10.0f;
float fireRate = 0.5f;
float currentGunScale = 0.008f;
glm::vec3 currentGunOffset(1.5f, 0.05f, 0.0f);
float gunRotateY = 0.0f;

// --- GEOMETRY ---
float planeVertices[] = {
     25.0f, -0.4f,  25.0f,  25.0f, 0.0f,
    -25.0f, -0.4f,  25.0f,   0.0f, 0.0f,
    -25.0f, -0.4f, -25.0f,   0.0f, 25.0f,
     25.0f, -0.4f,  25.0f,  25.0f, 0.0f,
    -25.0f, -0.4f, -25.0f,   0.0f, 25.0f,
     25.0f, -0.4f, -25.0f,  25.0f, 25.0f
};
float crosshairVertices[] = { -0.02f, 0.0f, 0.02f, 0.0f, 0.0f, -0.03f, 0.0f, 0.03f };
float cubeVertices[] = {
    -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f,
    -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f
};
float quadVertices[] = {
    -1.0f,  1.0f,
    -1.0f, -1.0f,
     1.0f, -1.0f,

    -1.0f,  1.0f,
     1.0f, -1.0f,
     1.0f,  1.0f
};

float skyboxVertices[] = {
    // positions          
    -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

    -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f
};

unsigned int loadCubemap(std::vector<std::string> faces)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(false);

    for (unsigned int i = 0; i < faces.size(); i++)
    {
        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data)
        {
            GLenum format = (nrChannels == 4) ? GL_RGBA : GL_RGB;
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }

    stbi_set_flip_vertically_on_load(true);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}

int main()
{
    srand(static_cast<unsigned int>(time(0)));
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "OrdinaryCold", NULL, NULL);
    if (window == NULL) { std::cout << "Failed to create window" << std::endl; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Start with cursor visible for the menu
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cout << "Failed to init GLAD" << std::endl; return -1; }
    stbi_set_flip_vertically_on_load(true);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // --- SHADERS SETUP ---
    const char* uiVS = "#version 330 core\nlayout (location = 0) in vec2 aPos;\nvoid main(){gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);}";
    const char* uiFS = "#version 330 core\nout vec4 FragColor;\nvoid main(){FragColor = vec4(0.0, 0.0, 0.0, 0.8);}";
    unsigned int uiProg = glCreateProgram();
    unsigned int vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vs, 1, &uiVS, NULL); glCompileShader(vs);
    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fs, 1, &uiFS, NULL); glCompileShader(fs);
    glAttachShader(uiProg, vs); glAttachShader(uiProg, fs); glLinkProgram(uiProg);
    glDeleteShader(vs); glDeleteShader(fs);

    const char* crossVS = "#version 330 core\nlayout (location = 0) in vec2 aPos;\nvoid main() { gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0); }";
    const char* crossFS = "#version 330 core\nout vec4 FragColor;\nvoid main() { FragColor = vec4(0.0, 1.0, 0.0, 1.0); }";
    unsigned int crossProg = glCreateProgram();
    vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vs, 1, &crossVS, NULL); glCompileShader(vs);
    fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fs, 1, &crossFS, NULL); glCompileShader(fs);
    glAttachShader(crossProg, vs); glAttachShader(crossProg, fs); glLinkProgram(crossProg);
    glDeleteShader(vs); glDeleteShader(fs);

    const char* gunVS = "#version 330 core\nlayout (location = 0) in vec3 aPos;\nuniform mat4 model, view, projection;\nvoid main() { gl_Position = projection * view * model * vec4(aPos, 1.0); }";
    const char* gunFS = "#version 330 core\nout vec4 FragColor;\nuniform vec4 color;\nvoid main() { FragColor = color; }";
    unsigned int gunProg = glCreateProgram();
    vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vs, 1, &gunVS, NULL); glCompileShader(vs);
    fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fs, 1, &gunFS, NULL); glCompileShader(fs);
    glAttachShader(gunProg, vs); glAttachShader(gunProg, fs); glLinkProgram(gunProg);
    glDeleteShader(vs); glDeleteShader(fs);

    const char* floorVS = "#version 330 core\nlayout (location = 0) in vec3 aPos;\nuniform mat4 model, view, projection;\nout vec3 FragPos;\nvoid main() { FragPos = vec3(model * vec4(aPos, 1.0)); gl_Position = projection * view * vec4(FragPos, 1.0); }";
    const char* floorFS = "#version 330 core\nout vec4 FragColor;\nin vec3 FragPos;\nvoid main() { float checkSize = 1.0; float f = mod(floor(FragPos.x / checkSize) + floor(FragPos.z / checkSize), 2.0); vec3 color = mix(vec3(0.2), vec3(0.4), f); FragColor = vec4(color, 1.0); }";
    unsigned int floorProg = glCreateProgram();
    vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vs, 1, &floorVS, NULL); glCompileShader(vs);
    fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fs, 1, &floorFS, NULL); glCompileShader(fs);
    glAttachShader(floorProg, vs); glAttachShader(floorProg, fs); glLinkProgram(floorProg);
    glDeleteShader(vs); glDeleteShader(fs);

    unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    const char* skyboxVS = "#version 330 core\nlayout (location = 0) in vec3 aPos;\nout vec3 TexCoords;\nuniform mat4 projection;\nuniform mat4 view;\nvoid main(){\nTexCoords = aPos;\nvec4 pos = projection * view * vec4(aPos, 1.0);\ngl_Position = pos.xyww;\n}";
    const char* skyboxFS = "#version 330 core\nout vec4 FragColor;\nin vec3 TexCoords;\nuniform samplerCube skybox;\nvoid main(){\nFragColor = texture(skybox, TexCoords);\n}";

    unsigned int skyboxProg = glCreateProgram();
    unsigned int svs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(svs, 1, &skyboxVS, NULL); glCompileShader(svs);
    unsigned int sfs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(sfs, 1, &skyboxFS, NULL); glCompileShader(sfs);
    glAttachShader(skyboxProg, svs); glAttachShader(skyboxProg, sfs); glLinkProgram(skyboxProg);
    glDeleteShader(svs); glDeleteShader(sfs);

    // 3. Load Textures
    std::vector<std::string> faces{
        FileSystem::getPath("resources/textures/skybox/right.jpg"),
        FileSystem::getPath("resources/textures/skybox/left.jpg"),
        FileSystem::getPath("resources/textures/skybox/top.jpg"),
        FileSystem::getPath("resources/textures/skybox/bottom.jpg"),
        FileSystem::getPath("resources/textures/skybox/front.jpg"),
        FileSystem::getPath("resources/textures/skybox/back.jpg")
    };
    unsigned int cubemapTexture = loadCubemap(faces);

    // 4. Configure Shader
    glUseProgram(skyboxProg);
    glUniform1i(glGetUniformLocation(skyboxProg, "skybox"), 0);

    // --- VAO/VBO SETUP ---
    unsigned int planeVAO, planeVBO;
    glGenVertexArrays(1, &planeVAO); glGenBuffers(1, &planeVBO); glBindVertexArray(planeVAO); glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);

    unsigned int crossVAO, crossVBO;
    glGenVertexArrays(1, &crossVAO); glGenBuffers(1, &crossVBO); glBindVertexArray(crossVAO); glBindBuffer(GL_ARRAY_BUFFER, crossVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(crosshairVertices), crosshairVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    unsigned int cubeVAO, cubeVBO;
    glGenVertexArrays(1, &cubeVAO); glGenBuffers(1, &cubeVBO); glBindVertexArray(cubeVAO); glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO); glBindVertexArray(quadVAO); glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // --- LOAD ASSETS ---
    Shader ourShader("anim_model.vs", "anim_model.fs");
    Model ourModel(FileSystem::getPath("resources/objects/mixamo/Ch49_nonPBR.dae"));

    // Load Animations into Globals
    Animation idleAnimation(FileSystem::getPath("resources/objects/mixamo/idle.dae"), &ourModel);
    Animation walkAnimation(FileSystem::getPath("resources/objects/mixamo/walk.dae"), &ourModel);
    Animation runAnimation(FileSystem::getPath("resources/objects/mixamo/run.dae"), &ourModel);
    Animation deathAnimation(FileSystem::getPath("resources/objects/mixamo/death.dae"), &ourModel);

    // Store pointers
    pIdleAnim = &idleAnimation;
    pWalkAnim = &walkAnimation;
    pRunAnim = &runAnimation;
    pDeathAnim = &deathAnimation;

    // Player Animator
    Animator playerAnimator(&walkAnimation);

    Model gunModel(FileSystem::getPath("resources/objects/mixamo/M4A1.dae"));
    glm::vec3 gmin(FLT_MAX), gmax(-FLT_MAX);
    for (auto& m : gunModel.meshes) {
        for (auto& v : m.vertices) { gmin = glm::min(gmin, v.Position); gmax = glm::max(gmax, v.Position); }
    }
    glm::vec3 gunCenter = (gmin + gmax) * 0.5f;

    CreateMap();
    ResetGame();

    std::cout << "========================================" << std::endl;
    std::cout << "   CLICK MOUSE 1 (LEFT) TO START GAME   " << std::endl;
    std::cout << "========================================" << std::endl;

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // --- START SCREEN LOGIC ---
        if (!gameStarted && !playerIsDead) {
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                gameStarted = true;
                firstMouse = true;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
        }

        // --- PLAYER DEATH LOGIC ---
        if (playerIsDead) {
            float deathDuration = pDeathAnim->GetDuration();
            float timeToAdvance = deltaTime;

            if (playerDeathTimer + timeToAdvance >= deathDuration) {
                timeToAdvance = deathDuration - playerDeathTimer - 0.001f;
            }
            if (timeToAdvance > 0) playerAnimator.UpdateAnimation(timeToAdvance);

            playerDeathTimer += deltaTime;
            if (playerDeathTimer > 3.0f) {
                ResetGame();
            }
        }

        bool isMoving = false;
        bool isRunning = false;

        // --- GAME LOOP ---
        if (gameStarted || playerIsDead) {

            if (!playerIsDead) {
                isMoving = processInput(window);
                isRunning = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);
            }

            // Time Logic
            bool inRecoil = (currentFrame - lastShotTime < fireRate);
            float targetTimeScale = 0.0f;

            if (playerIsDead) targetTimeScale = 0.0f;
            else if (isMoving) targetTimeScale = 1.0f;
            else if (inRecoil) targetTimeScale = 0.3f;
            else if (mouseMoveAmt > 1.5f || glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) targetTimeScale = 0.2f;

            timeScale = glm::mix(timeScale, targetTimeScale, 15.0f * deltaTime);
            mouseMoveAmt = 0.0f;

            if (!playerIsDead) ManageWaves(deltaTime * timeScale);

            if (!playerIsDead && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                if (currentFrame - lastShotTime > fireRate) {
                    ShootPlayer(playerPos);
                    lastShotTime = currentFrame;
                }
            }

            // Debug Gun keys
            if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) gunRotateY = 180.0f;
            if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) gunRotateY = 0.0f;

            // --- PLAYER ANIMATION ---
            if (playerIsDead) {
                if (playerAnimator.m_CurrentAnimation != pDeathAnim)
                    playerAnimator.PlayAnimation(pDeathAnim, NULL, 0.0f, 0.0f, 0.0f);
            }
            else {
                if (isRunning) { if (playerAnimator.m_CurrentAnimation != pRunAnim) playerAnimator.PlayAnimation(pRunAnim, NULL, playerAnimator.m_CurrentTime, 0.0f, 0.0f); }
                else { if (playerAnimator.m_CurrentAnimation != pWalkAnim) playerAnimator.PlayAnimation(pWalkAnim, NULL, playerAnimator.m_CurrentTime, 0.0f, 0.0f); }

                if (isMoving) playerAnimator.UpdateAnimation(deltaTime);
                else playerAnimator.UpdateAnimation(0.0f);
            }

            // --- ENEMY LOGIC ---
            for (auto& enemy : enemies) {
                if (enemy.isDead) {
                    if (enemy.animator.m_CurrentAnimation != pDeathAnim)
                        enemy.animator.PlayAnimation(pDeathAnim, NULL, 0.0f, 0.0f, 0.0f);

                    float deathDuration = pDeathAnim->GetDuration();
                    float advanceTime = deltaTime * timeScale;

                    if (timeScale > 0.01f) {
                        if (enemy.deadTimeCounter + advanceTime >= deathDuration) {
                            advanceTime = deathDuration - enemy.deadTimeCounter - 0.001f;
                        }

                        if (advanceTime > 0.0f) {
                            enemy.animator.UpdateAnimation(advanceTime);
                        }
                    }

                    enemy.deadTimeCounter += deltaTime * timeScale;
                    continue;
                }

                if (playerIsDead) {
                    enemy.animator.UpdateAnimation(0.0f);
                    continue;
                }

                if (enemy.animator.m_CurrentAnimation != pRunAnim) enemy.animator.PlayAnimation(pRunAnim, NULL, enemy.animator.m_CurrentTime, 0.0f, 0.0f);
                if (timeScale > 0.05f) enemy.animator.UpdateAnimation(deltaTime * timeScale); else enemy.animator.UpdateAnimation(0.0f);

                float distToPlayer = glm::distance(enemy.position, playerPos);
                if (distToPlayer < 1.5f) {
                    TriggerPlayerDeath();
                }

                glm::vec3 bestDir = GetSmartPath(enemy.position, playerPos);
                enemy.position = MoveWithSliding(enemy.position, bestDir * enemy.speed * deltaTime * timeScale, ENEMY_HITBOX_RADIUS);

                enemy.shootTimer -= deltaTime * timeScale;
                if (enemy.shootTimer <= 0.0f) {
                    glm::vec3 gunPos = enemy.position; gunPos.y += 1.3f;
                    ShootEnemy(enemy, playerPos, gunPos);
                    enemy.shootTimer = enemy.shootInterval;
                }
            }

            enemies.erase(std::remove_if(enemies.begin(), enemies.end(), [](const Enemy& e) {
                return e.isDead && e.deadTimeCounter > 2.5f;
                }), enemies.end());

            // --- BULLET LOGIC ---
            for (int i = 0; i < bullets.size(); i++) {
                float dist = bullets[i].speed * deltaTime * timeScale;
                glm::vec3 nxt = bullets[i].position + bullets[i].direction * dist;
                bool hit = false;
                if (IsCollidingWithMap(nxt, 0.1f)) hit = true;

                if (!hit) {
                    if (bullets[i].isPlayerBullet) {
                        for (auto& e : enemies) {
                            if (e.isDead) continue; 
                            glm::vec3 box = e.position; box.y += 1.0f;
                            float d = GetRaySphereIntersection(bullets[i].position, bullets[i].direction, box, ENEMY_HITBOX_RADIUS);
                            if (d >= 0.0f && d <= dist) {
                                e.hp = 0;
                                e.isDead = true;
                                hit = true;
                                break;
                            }
                        }
                    }
                    else {
                        if (!playerIsDead) {
                            glm::vec3 box = playerPos; box.y += 1.0f;
                            float d = GetRaySphereIntersection(bullets[i].position, bullets[i].direction, box, 0.6f);
                            if (d >= 0.0f && d <= dist) {
                                TriggerPlayerDeath();
                                hit = true;
                            }
                        }
                    }
                }
                if (hit) bullets[i].life = -1; else bullets[i].position = nxt;
                bullets[i].life -= deltaTime * timeScale;
            }
            bullets.erase(std::remove_if(bullets.begin(), bullets.end(), [](const Bullet& b) { return b.life <= 0; }), bullets.end());
        }

    Render:
        // --- DRAW ---
        float camDist = 0.6f, camHeight = 0.8f, rightOffset = 0.3f;

        glm::vec3 flatFront = glm::normalize(glm::vec3(camera.Front.x, 0.0f, camera.Front.z));
        glm::vec3 flatRight = glm::normalize(glm::cross(flatFront, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 cPos = playerPos - flatFront * camDist + flatRight * rightOffset; cPos.y += camHeight;

        if (playerIsDead) {
            cPos.y = glm::max(0.1f, cPos.y - (playerDeathTimer * 0.5f));
        }
        camera.Position = cPos;

        glClearColor(0.05f, 0.05f, 0.05f, 1.0f); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glm::mat4 proj = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();

        // Floor
        glUseProgram(floorProg);
        glUniformMatrix4fv(glGetUniformLocation(floorProg, "projection"), 1, GL_FALSE, &proj[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(floorProg, "view"), 1, GL_FALSE, &view[0][0]);
        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(floorProg, "model"), 1, GL_FALSE, &model[0][0]);
        glBindVertexArray(planeVAO); glDrawArrays(GL_TRIANGLES, 0, 6);

        // Walls
        glUseProgram(gunProg);
        glUniform4f(glGetUniformLocation(gunProg, "color"), 0.8f, 0.8f, 0.8f, 1.0f);
        glUniformMatrix4fv(glGetUniformLocation(gunProg, "projection"), 1, GL_FALSE, &proj[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(gunProg, "view"), 1, GL_FALSE, &view[0][0]);
        glBindVertexArray(cubeVAO);
        for (auto& w : mapObstacles) {
            model = glm::translate(glm::mat4(1.0f), w.position); model = glm::scale(model, w.size);
            glUniformMatrix4fv(glGetUniformLocation(gunProg, "model"), 1, GL_FALSE, &model[0][0]);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        // Player
        ourShader.use(); ourShader.setMat4("projection", proj); ourShader.setMat4("view", view);
        auto pBone = playerAnimator.GetFinalBoneMatrices();
        for (int i = 0; i < pBone.size(); ++i) ourShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", pBone[i]);
        model = glm::translate(glm::mat4(1.0f), playerPos);
        model = glm::rotate(model, atan2(camera.Front.x, camera.Front.z), glm::vec3(0, 1, 0));
        model = glm::scale(model, glm::vec3(0.5f));
        ourShader.setMat4("model", model);
        ourModel.Draw(ourShader);

        // Player Gun
        auto boneMap = idleAnimation.GetBoneIDMap();
        if (boneMap.find("mixamorig_RightHand") != boneMap.end()) {
            int id = boneMap["mixamorig_RightHand"].id;
            glm::mat4 bone = pBone[id] * glm::inverse(boneMap["mixamorig_RightHand"].offset);
            glm::mat4 attach = model * bone * glm::translate(glm::mat4(1.0f), currentGunOffset);
            currentGunBarrelPos = glm::vec3(attach[3]);

            glm::vec3 aim = camera.Position + camera.Front * 100.0f;
            if (playerIsDead) aim = currentGunBarrelPos + glm::vec3(0, -1, 0);

            if (!playerIsDead) {
                float closest = 100.0f;
                for (auto& w : mapObstacles) {
                    float d = GetRayAABBIntersection(camera.Position, camera.Front, w);
                    if (d > 0 && d < closest) { closest = d; aim = camera.Position + camera.Front * d; }
                }
                for (auto& e : enemies) {
                    if (e.isDead) continue;
                    glm::vec3 box = e.position; box.y += 1.0f;
                    float d = GetRaySphereIntersection(camera.Position, camera.Front, box, ENEMY_HITBOX_RADIUS);
                    if (d > 0 && d < closest) { closest = d; aim = camera.Position + camera.Front * d; }
                }
                if (closest < 3.0f) {
                    aim = camera.Position + camera.Front * 100.0f;
                }
            }
            glm::vec3 aimDir = aim - currentGunBarrelPos;
            if (!playerIsDead) {
                aimDir.y = 0.0f; 
            }
            glm::quat q = glm::quatLookAt(glm::normalize(aim - currentGunBarrelPos), glm::vec3(0, 1, 0));
            glm::mat4 gModel = glm::translate(glm::mat4(1.0f), currentGunBarrelPos) * glm::toMat4(q);
            gModel = glm::rotate(gModel, glm::radians(gunRotateY), glm::vec3(0, 1, 0));
            gModel = glm::scale(gModel, glm::vec3(currentGunScale));
            gModel = glm::translate(gModel, -gunCenter);

            glUseProgram(gunProg); glUniform4f(glGetUniformLocation(gunProg, "color"), 0.2f, 0.2f, 0.25f, 1.0f);
            glUniformMatrix4fv(glGetUniformLocation(gunProg, "projection"), 1, GL_FALSE, &proj[0][0]);
            glUniformMatrix4fv(glGetUniformLocation(gunProg, "view"), 1, GL_FALSE, &view[0][0]);
            glUniformMatrix4fv(glGetUniformLocation(gunProg, "model"), 1, GL_FALSE, &gModel[0][0]);
            gunModel.Draw(ourShader);
        }

        // Enemies
        ourShader.use();
        for (auto& e : enemies) {
            auto eBones = e.animator.GetFinalBoneMatrices();
            for (int i = 0; i < eBones.size(); ++i) ourShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", eBones[i]);

            model = glm::translate(glm::mat4(1.0f), e.position);
            glm::vec3 dir = glm::normalize(playerPos - e.position);
            model = glm::rotate(model, atan2(dir.x, dir.z), glm::vec3(0, 1, 0));
            model = glm::scale(model, glm::vec3(0.5f));
            ourShader.setMat4("model", model);
            ourModel.Draw(ourShader);

            if (!e.isDead && boneMap.find("mixamorig_RightHand") != boneMap.end()) {
                int id = boneMap["mixamorig_RightHand"].id;
                glm::mat4 bone = eBones[id] * glm::inverse(boneMap["mixamorig_RightHand"].offset);
                glm::mat4 attach = model * bone * glm::translate(glm::mat4(1.0f), currentGunOffset);
                glm::vec3 gPos = glm::vec3(attach[3]);
                glm::vec3 t = playerPos; t.y += 0.8f;
                glm::quat q = glm::quatLookAt(glm::normalize(t - gPos), glm::vec3(0, 1, 0));

                glm::mat4 gModel = glm::translate(glm::mat4(1.0f), gPos) * glm::toMat4(q);
                gModel = glm::translate(gModel, glm::vec3(0.0f, 0.0f, -0.2f));
                gModel = glm::rotate(gModel, glm::radians(gunRotateY), glm::vec3(0, 1, 0));
                gModel = glm::scale(gModel, glm::vec3(currentGunScale));
                gModel = glm::translate(gModel, -gunCenter);

                glUseProgram(gunProg); glUniform4f(glGetUniformLocation(gunProg, "color"), 0.8f, 0.1f, 0.1f, 1.0f);
                glUniformMatrix4fv(glGetUniformLocation(gunProg, "projection"), 1, GL_FALSE, &proj[0][0]);
                glUniformMatrix4fv(glGetUniformLocation(gunProg, "view"), 1, GL_FALSE, &view[0][0]);
                glUniformMatrix4fv(glGetUniformLocation(gunProg, "model"), 1, GL_FALSE, &gModel[0][0]);
                gunModel.Draw(ourShader);
                ourShader.use();
            }
        }

        // Bullets
        glUseProgram(gunProg);
        glUniformMatrix4fv(glGetUniformLocation(gunProg, "projection"), 1, GL_FALSE, &proj[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(gunProg, "view"), 1, GL_FALSE, &view[0][0]);
        glBindVertexArray(cubeVAO);
        for (auto& b : bullets) {
            if (b.isPlayerBullet) glUniform4f(glGetUniformLocation(gunProg, "color"), 1.0f, 0.8f, 0.0f, 1.0f);
            else glUniform4f(glGetUniformLocation(gunProg, "color"), 1.0f, 0.0f, 0.0f, 1.0f);
            model = glm::translate(glm::mat4(1.0f), b.position); model = glm::scale(model, glm::vec3(0.05f));
            glUniformMatrix4fv(glGetUniformLocation(gunProg, "model"), 1, GL_FALSE, &model[0][0]);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        // --- DRAW SKYBOX ---
        glDepthFunc(GL_LEQUAL); 
        glUseProgram(skyboxProg);

        glm::mat4 skyView = glm::mat4(glm::mat3(camera.GetViewMatrix()));

        glUniformMatrix4fv(glGetUniformLocation(skyboxProg, "view"), 1, GL_FALSE, &skyView[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(skyboxProg, "projection"), 1, GL_FALSE, &proj[0][0]);

        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS); 

        glDisable(GL_DEPTH_TEST);
        glUseProgram(crossProg);
        glBindVertexArray(crossVAO); glDrawArrays(GL_LINES, 0, 4);

        if (!gameStarted && !playerIsDead) {
            glEnable(GL_BLEND); glUseProgram(uiProg);
            glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6);
            glDisable(GL_BLEND);
        }

        glEnable(GL_DEPTH_TEST);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate(); return 0;
}

// --- IMPLEMENTATIONS ---

void CreateMap() {
    mapObstacles.clear();
    Obstacle w; w.size = glm::vec3(50, 5, 2);
    w.position = glm::vec3(0, 2, -25); mapObstacles.push_back(w);
    w.position = glm::vec3(0, 2, 25); mapObstacles.push_back(w);
    w.size = glm::vec3(2, 5, 50);
    w.position = glm::vec3(-25, 2, 0); mapObstacles.push_back(w);
    w.position = glm::vec3(25, 2, 0); mapObstacles.push_back(w);
    w.size = glm::vec3(4, 5, 4);
    w.position = glm::vec3(10, 2, 10); mapObstacles.push_back(w);
    w.position = glm::vec3(-10, 2, -10); mapObstacles.push_back(w);
    w.size = glm::vec3(8, 3, 1);
    w.position = glm::vec3(-5, 1.5, 8); mapObstacles.push_back(w);
}

bool processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
    bool mv = false;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) playerSpeed = 5.0f; else playerSpeed = 2.5f;
    glm::vec3 f = glm::normalize(glm::vec3(camera.Front.x, 0, camera.Front.z));
    glm::vec3 r = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));
    glm::vec3 n = playerPos;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { n += f * playerSpeed * deltaTime; mv = true; }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { n -= f * playerSpeed * deltaTime; mv = true; }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { n -= r * playerSpeed * deltaTime; mv = true; }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { n += r * playerSpeed * deltaTime; mv = true; }
    if (!IsCollidingWithMap(n, playerRadius)) playerPos = n;
    return mv;
}

void ManageWaves(float dt) {
    if (enemies.empty() && enemiesSpawnedInWave >= enemiesToSpawnTotal) {
        currentWave++;
        enemiesToSpawnTotal = 3 + ((currentWave - 1) / 3);
        enemiesSpawnedInWave = 0;
        std::cout << "Wave " << currentWave << " Start!" << std::endl;
    }
    if (enemiesSpawnedInWave < enemiesToSpawnTotal) SpawnEnemyInFOV();
}

void SpawnEnemyInFOV() {
    for (int i = 0; i < 10; i++) {
        float ang = (float)(rand() % 360);
        float rad = glm::radians(ang);
        float dist = ENEMY_SPAWN_RADIUS_MIN + (float)(rand() % 100) / 100.0f * (ENEMY_SPAWN_RADIUS_MAX - ENEMY_SPAWN_RADIUS_MIN);
        glm::vec3 pos(playerPos.x + dist * cos(rad), -0.4f, playerPos.z + dist * sin(rad));
        if (!IsCollidingWithMap(pos, ENEMY_HITBOX_RADIUS) && IsInsideFOV(pos, camera, camera.Zoom)) {
            Enemy e; e.position = pos; e.hp = ENEMY_HP; e.speed = ENEMY_SPEED;
            e.shootInterval = 1.0f + (float)(rand() % 200) / 100.0f; e.shootTimer = e.shootInterval;
            e.animator = Animator(pRunAnim);
            e.isDead = false; e.deadTimeCounter = 0.0f;
            enemies.push_back(e);
            enemiesSpawnedInWave++;
            break;
        }
    }
}

void ShootPlayer(glm::vec3 startPos) {
    glm::vec3 s = (currentGunBarrelPos == glm::vec3(0)) ? camera.Position : currentGunBarrelPos;
    glm::vec3 dir = camera.Front; dir.y = 0; dir = glm::normalize(dir);
    if (glm::length(dir) < 0.01f) dir = glm::vec3(camera.Front.x, 0, camera.Front.z);
    Bullet b; b.position = s; b.direction = dir; b.speed = 5.0f; b.life = 3.0f; b.isPlayerBullet = true;
    bullets.push_back(b);
}

void ShootEnemy(Enemy& e, glm::vec3 pPos, glm::vec3 gPos) {
    Bullet b; b.position = gPos; b.position.y -= 0.6f;
    glm::vec3 dir = pPos - b.position; dir.y = 0;
    b.direction = glm::normalize(dir); b.speed = ENEMY_BULLET_SPEED; b.life = 5.0f; b.isPlayerBullet = false;
    bullets.push_back(b);
}

void TriggerPlayerDeath() {
    if (playerIsDead) return;
    std::cout << "PLAYER DIED!" << std::endl;
    playerIsDead = true;
    playerDeathTimer = 0.0f;
}

void ResetGame() {
    enemies.clear(); bullets.clear();
    playerPos = glm::vec3(0, -0.4f, 0);
    currentWave = 1; enemiesToSpawnTotal = 3; enemiesSpawnedInWave = 0;
    gameStarted = false;
    playerIsDead = false;
    playerDeathTimer = 0.0f;
    glfwSetInputMode(glfwGetCurrentContext(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    std::cout << "RESET GAME - Click to Retry" << std::endl;
}

void framebuffer_size_callback(GLFWwindow* w, int width, int height) { glViewport(0, 0, width, height); }
void mouse_callback(GLFWwindow* w, double x, double y) {
    if (!gameStarted || playerIsDead) return;
    if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }
    float xo = x - lastX; float yo = lastY - y;
    mouseMoveAmt += sqrt(xo * xo + yo * yo);
    camera.ProcessMouseMovement(xo, yo);
    lastX = x; lastY = y;
}
void scroll_callback(GLFWwindow* w, double x, double y) { camera.ProcessMouseScroll(y); }