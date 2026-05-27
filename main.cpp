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
#include "D:\Uni Works\Year3\Advcg1_Assi3\AdvCG1_Assi3\src\8.guest\2020\skeletal_animation\WaterSurface.h"
#include "D:\Uni Works\Year3\Advcg1_Assi3\AdvCG1_Assi3\src\8.guest\2020\skeletal_animation\FireSim.h"

#include <iostream>
#include <stb_image.h>

// Include our new modular headers
#include "Globals.h"
#include "Physics.h"
#include "GameManager.h"
#include "Input.h"

// =========================================================
// GLOBAL VARIABLE DEFINITIONS (Allocating memory for externs)
// =========================================================
const unsigned int SCR_WIDTH = 1600;
const unsigned int SCR_HEIGHT = 800;
Camera camera(glm::vec3(0.0f, 0.0f, 5.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

glm::vec3 playerPos(-22.382f, -0.4f, 11.1701f);
float playerSpeed = 2.5f;
float playerRadius = 0.5f;
bool playerIsDead = false;
float playerDeathTimer = 0.0f;

Animation* pIdleAnim = nullptr;
Animation* pWalkW = nullptr;
Animation* pWalkS = nullptr;
Animation* pStrafeA = nullptr;
Animation* pStrafeD = nullptr;
Animation* pDeathAnim = nullptr;
Animation* pRunAnim = nullptr;
Animation* currentMovementAnim = nullptr;
// (Replace the old wave/enemy globals with these)
Animation* pPlayerSlash = nullptr;
Animation* pBossJumpAttack = nullptr;

bool playerIsAttacking = false;
Boss activeBoss;

bool gameStarted = false;
float ENEMY_HITBOX_RADIUS = 0.8f;

std::vector<Obstacle> mapObstacles;
// =========================================================


// --- GEOMETRY DATA ---
float cubeVertices[] = {
    -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f,
    -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f
};

float skyboxVertices[] = {
    -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f
};

unsigned int loadCubemap(std::vector<std::string> faces) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(false);
    for (unsigned int i = 0; i < faces.size(); i++) {
        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            GLenum format = (nrChannels == 4) ? GL_RGBA : GL_RGB;
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else {
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
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cout << "Failed to init GLAD" << std::endl; return -1; }

    WaterSurface water(50.0f, 50.0f, -0.4f, 0.2f);

    stbi_set_flip_vertically_on_load(true);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    // (Vertex Shader remains the same: High Frequency Micro-Ripples)
    const char* waterVS = "#version 330 core\nlayout (location = 0) in vec3 aPos;\nlayout (location = 1) in vec3 aNormal;\nuniform mat4 model, view, projection;\nuniform float time;\nout vec3 FragPos, Normal;\nvoid main() { vec3 pos = aPos; float wave1 = sin(pos.x * 12.0 + time * 2.0) * cos(pos.z * 10.0 + time * 1.8) * 0.005; float wave2 = sin(pos.x * 20.0 - time * 2.5) * 0.003; pos.y += wave1 + wave2; FragPos = vec3(model * vec4(pos, 1.0)); vec3 newNormal = aNormal + vec3(cos(pos.x * 12.0 + time * 2.0) * 0.12 + cos(pos.x * 20.0 - time * 2.5) * 0.12, 0.0, -sin(pos.z * 10.0 + time * 1.8) * 0.12); Normal = mat3(transpose(inverse(model))) * normalize(newNormal); gl_Position = projection * view * vec4(FragPos, 1.0); }";
    // (NEW Fragment Shader: Blood Vanguard style: More Red, Less Reflective)
    const char* waterFS = "#version 330 core\nout vec4 FragColor;\nin vec3 FragPos, Normal;\nuniform vec3 viewPos;\nuniform samplerCube skybox;\nvoid main() { vec3 norm = normalize(Normal); vec3 viewDir = normalize(viewPos - FragPos); vec3 I = normalize(FragPos - viewPos); vec3 reflectDir = reflect(I, norm); vec3 reflection = texture(skybox, reflectDir).rgb; vec3 bloodColor = vec3(0.12, 0.0, 0.0); float fresnel = pow(1.0 - max(dot(norm, viewDir), 0.0), 4.0); vec3 finalColor = mix(bloodColor, reflection, 0.05 + 0.25 * fresnel); vec3 lightDir = normalize(vec3(10.0, 50.0, 10.0)); vec3 halfDir = normalize(lightDir + viewDir); float spec = pow(max(dot(norm, halfDir), 0.0), 128.0); finalColor += vec3(0.6, 0.2, 0.2) * spec * 0.4; FragColor = vec4(finalColor, 0.98); }";
    // Remember the previously restored missing line:
    unsigned int waterProg = glCreateProgram();
    unsigned int wvs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(wvs, 1, &waterVS, NULL); glCompileShader(wvs);
    unsigned int wfs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(wfs, 1, &waterFS, NULL); glCompileShader(wfs);
    glAttachShader(waterProg, wvs); glAttachShader(waterProg, wfs); glLinkProgram(waterProg);
    glDeleteShader(wvs); glDeleteShader(wfs);

    // --- FIRE BILLBOARD SHADERS ---
    const char* fireVS = "#version 330 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "layout (location = 1) in vec2 aTexCoords;\n"
        "out vec2 TexCoords;\n"
        "uniform mat4 projection, view;\n"
        "uniform vec3 centerPos;\n"
        "uniform vec2 size;\n"
        "uniform vec3 cameraRight, cameraUp;\n"
        "void main() {\n"
        "    TexCoords = aTexCoords;\n"
        "    vec3 vertexPos = centerPos + cameraRight * aPos.x * size.x + cameraUp * aPos.y * size.y;\n"
        "    gl_Position = projection * view * vec4(vertexPos, 1.0);\n"
        "}\n";

    const char* fireFS = "#version 330 core\n"
        "out vec4 FragColor;\n"
        "in vec2 TexCoords;\n"
        "uniform sampler2D fireTexture;\n"
        "void main() {\n"
        "    vec4 color = texture(fireTexture, TexCoords);\n"
        "    // If Red isn't significantly higher than Blue, it's smoke or background. Discard it!\n"
        "    if(color.r <= color.b + 0.1) discard;\n"
        "    FragColor = vec4(color.rgb, 1.0);\n"
        "}\n";
    unsigned int fireProg = glCreateProgram();
    unsigned int fvs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(fvs, 1, &fireVS, NULL); glCompileShader(fvs);
    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fs, 1, &fireFS, NULL); glCompileShader(fs);
    glAttachShader(fireProg, fvs); glAttachShader(fireProg, fs); glLinkProgram(fireProg);
    glDeleteShader(fvs); glDeleteShader(fs);

    // (Skybox Shader & VAO)
    unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO); glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO); glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    const char* skyboxVS = "#version 330 core\nlayout (location = 0) in vec3 aPos;\nout vec3 TexCoords;\nuniform mat4 projection, view;\nvoid main(){ TexCoords = aPos; vec4 pos = projection * view * vec4(aPos, 1.0); gl_Position = pos.xyww; }";
    const char* skyboxFS = "#version 330 core\nout vec4 FragColor;\nin vec3 TexCoords;\nuniform samplerCube skybox;\nvoid main(){ FragColor = texture(skybox, TexCoords); }";
    unsigned int skyboxProg = glCreateProgram();
    unsigned int svs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(svs, 1, &skyboxVS, NULL); glCompileShader(svs);
    unsigned int sfs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(sfs, 1, &skyboxFS, NULL); glCompileShader(sfs);
    glAttachShader(skyboxProg, svs); glAttachShader(skyboxProg, sfs); glLinkProgram(skyboxProg);
    glDeleteShader(svs); glDeleteShader(sfs);

    std::vector<std::string> faces{
        FileSystem::getPath("resources/textures/skybox/right.jpg"),
        FileSystem::getPath("resources/textures/skybox/left.jpg"),
        FileSystem::getPath("resources/textures/skybox/top.jpg"),
        FileSystem::getPath("resources/textures/skybox/bottom.jpg"),
        FileSystem::getPath("resources/textures/skybox/front.jpg"),
        FileSystem::getPath("resources/textures/skybox/back.jpg")
    };
    unsigned int cubemapTexture = loadCubemap(faces);
    glUseProgram(skyboxProg);
    glUniform1i(glGetUniformLocation(skyboxProg, "skybox"), 0);

    // Map Cube VAO
    unsigned int cubeVAO, cubeVBO;
    glGenVertexArrays(1, &cubeVAO); glGenBuffers(1, &cubeVBO); glBindVertexArray(cubeVAO); glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    // --- LOAD ASSETS ---
    Shader ourShader("src/8.guest/2020/skeletal_animation/anim_model.vs", "src/8.guest/2020/skeletal_animation/anim_model.fs");
    Model ourModel(FileSystem::getPath("resources/objects/Paladin/Paladin.dae"));
    
    Model bossModel(FileSystem::getPath("resources/objects/Boss/Boss.dae"));
    Animation bossJumpAnim(FileSystem::getPath("resources/objects/Boss/JumpAttack.dae"), &bossModel);
    pBossJumpAttack = &bossJumpAnim;
    activeBoss.animator = Animator(&bossJumpAnim);

    Shader staticShader("src/8.guest/2020/skeletal_animation/static_model.vs", "src/8.guest/2020/skeletal_animation/static_model.fs");
    Model mapModel(FileSystem::getPath("resources/objects/map/source/Kakariko Village.fbx"));

    Animation idleAnimation(FileSystem::getPath("resources/objects/Paladin/Idle.dae"), &ourModel);
    Animation walkWAnim(FileSystem::getPath("resources/objects/Paladin/WalkW.dae"), &ourModel);
    Animation walkSAnim(FileSystem::getPath("resources/objects/Paladin/WalkS.dae"), &ourModel);
    Animation strafeAAnim(FileSystem::getPath("resources/objects/Paladin/StrafeA.dae"), &ourModel);
    Animation strafeDAnim(FileSystem::getPath("resources/objects/Paladin/StrafeD.dae"), &ourModel);
    Animation deathAnimation(FileSystem::getPath("resources/objects/Paladin/Death.dae"), &ourModel);
    Animation runAnimation(FileSystem::getPath("resources/objects/Paladin/Run.dae"), &ourModel);
    Animation slashAnimation(FileSystem::getPath("resources/objects/Paladin/Slash.dae"), &ourModel);

    pIdleAnim = &idleAnimation;
    pWalkW = &walkWAnim;
    pWalkS = &walkSAnim;
    pStrafeA = &strafeAAnim;
    pStrafeD = &strafeDAnim;
    pDeathAnim = &deathAnimation;
    pRunAnim = &runAnimation;
    pPlayerSlash = &slashAnimation;

    Animator playerAnimator(&idleAnimation);
    currentMovementAnim = pIdleAnim;

    CreateMap();
    ResetGame();

    std::cout << "========================================" << std::endl;
    std::cout << "   CLICK MOUSE 1 (LEFT) TO START GAME   " << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. Initialize CPU Fluid Simulator
    int numCells = 20000;
    float simHeight = 1.0f;
    float cScale = SCR_HEIGHT / simHeight;
    float simWidth = SCR_WIDTH / cScale;
    float fh = std::sqrt(simWidth * simHeight / numCells);
    int numX = std::floor(simWidth / fh);
    int numY = std::floor(simHeight / fh);

    Fluid* fluidSim = new Fluid(numX, numY, fh);
    
    // --- UPDATED SPAWN LOGIC ---
    fluidSim->obstacleX = 0.5f * numX * fh;
    fluidSim->obstacleY = 0.05f * numY * fh; // Spawn near the very bottom edge!
    fluidSim->obstacleRadius = 0.8f;         // Make the line extremely wide!

    // --- ADD THESE TWO LINES ---
    fluidSim->burningObstacle = false; // Turn off the fireball obstacle
    fluidSim->burningFloor = true;     // Turn on the straight horizontal wall of fire
    // ---------------------------

    std::vector<unsigned char> firePixelData(fluidSim->numX* fluidSim->numY * 3, 0);

    // 2. Setup Billboard Quad Geometry (Centered at 0,0)
    float fireVertices[] = {
         0.5f,  1.0f, 0.0f, 1.0f, 1.0f, // top right (Y is 1.0)
         0.5f,  0.0f, 0.0f, 1.0f, 0.0f, // bottom right (Y is 0.0)
        -0.5f,  0.0f, 0.0f, 0.0f, 0.0f, // bottom left (Y is 0.0)
        -0.5f,  1.0f, 0.0f, 0.0f, 1.0f  // top left (Y is 1.0)
    };
    unsigned int fireIndices[] = { 0, 1, 3, 1, 2, 3 };

    unsigned int fireVAO, fireVBO, fireEBO;
    glGenVertexArrays(1, &fireVAO); glGenBuffers(1, &fireVBO); glGenBuffers(1, &fireEBO);
    glBindVertexArray(fireVAO);
    glBindBuffer(GL_ARRAY_BUFFER, fireVBO); glBufferData(GL_ARRAY_BUFFER, sizeof(fireVertices), fireVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fireEBO); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(fireIndices), fireIndices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);

    // 3. Setup OpenGL Texture
    unsigned int fireTextureID;
    glGenTextures(1, &fireTextureID);
    glBindTexture(GL_TEXTURE_2D, fireTextureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // Allocate empty texture on GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fluidSim->numX, fluidSim->numY, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    std::vector<glm::vec3> firePositions = {
        glm::vec3(5.85698f, -0.4f,10.5844f),
        glm::vec3(0.334684f,  -0.4f, 15.6047f),
        glm::vec3(-28.6574f,  0.0f,  15.27f),
        glm::vec3(-28.3227f,  0.0f,  9.95687f)
    };

    bool isBirdsEyeView = false;
    bool spacePressed = false;
    bool leftMousePressed = false;

    while (!glfwWindowShouldClose(window))
    {

        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // --- INPUT & RAYCASTING LOGIC ---
        // Spacebar Toggle for Bird's-Eye View
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            if (!spacePressed) {
                isBirdsEyeView = !isBirdsEyeView;
                spacePressed = true;
            }
        }
        else {
            spacePressed = false;
        }

        // Left Click Logic
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (!leftMousePressed) {
                leftMousePressed = true;

                if (!gameStarted && !playerIsDead) {
                    gameStarted = true;
                    firstMouse = true;
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                }
                else if (gameStarted && !playerIsDead) {
                    // 1. Attack
                    if (!playerIsAttacking) playerIsAttacking = true;
                }
            }
        }
        else {
            leftMousePressed = false;
        }

        if (playerIsDead) {
            float deathDuration = pDeathAnim->GetDuration();
            float timeToAdvance = deltaTime;
            if (playerDeathTimer + timeToAdvance >= deathDuration) {
                timeToAdvance = deathDuration - playerDeathTimer - 0.001f;
            }
            if (timeToAdvance > 0) playerAnimator.UpdateAnimation(timeToAdvance);

            playerDeathTimer += deltaTime;
            if (playerDeathTimer > 3.0f) ResetGame();
        }

        if (gameStarted || playerIsDead) {
            if (!playerIsDead) {
                processInput(window);
                if (currentMovementAnim != pIdleAnim) water.AddRipple(playerPos, 0.8f, 0.005f);
            }

            water.Simulate(deltaTime);
            if (!playerIsDead) ManageWaves(deltaTime);

            if (playerIsDead) {
                if (playerAnimator.m_CurrentAnimation != pDeathAnim)
                    playerAnimator.PlayAnimation(pDeathAnim, NULL, 0.0f, 0.0f, 0.0f);
                playerAnimator.UpdateAnimation(deltaTime);
            }
            else if (playerIsAttacking) {
                // Play Slash Animation
                if (playerAnimator.m_CurrentAnimation != pPlayerSlash) {
                    playerAnimator.PlayAnimation(pPlayerSlash, NULL, 0.0f, 0.0f, 0.0f);
                }
                playerAnimator.UpdateAnimation(deltaTime);

                // End attack when animation finishes
                if (playerAnimator.m_CurrentTime >= pPlayerSlash->GetDuration() - 0.05f) {
                    playerIsAttacking = false;
                }
            }
            else {
                // Normal Movement
                if (playerAnimator.m_CurrentAnimation != currentMovementAnim) {
                    playerAnimator.PlayAnimation(currentMovementAnim, NULL, playerAnimator.m_CurrentTime, 0.0f, 0.0f);
                }
                playerAnimator.UpdateAnimation(deltaTime);
            }

            // --- BOSS AI & ANIMATION ---
            if (!activeBoss.isDead && !playerIsDead) {
                float distToPlayer = glm::distance(activeBoss.position, playerPos);

                if (activeBoss.state == 0) { // IDLE STATE (Used to be Chasing)

                    // --- MOVEMENT MATH HAS BEEN REMOVED ---

                    // Pause animator on frame 0 to simulate a static pose 
                    activeBoss.animator.UpdateAnimation(0.0f);

                    // If close enough, trigger Jump Attack!
                    if (distToPlayer < 4.5f) {
                        activeBoss.state = 1;
                        activeBoss.hasSlammed = false;
                        activeBoss.animator.PlayAnimation(pBossJumpAttack, NULL, 0.0f, 0.0f, 0.0f);
                    }
                }
                else if (activeBoss.state == 1) { // JUMP ATTACK STATE
                    activeBoss.animator.UpdateAnimation(deltaTime);
                    float animTime = activeBoss.animator.m_CurrentTime;
                    float animDuration = pBossJumpAttack->GetDuration();

                    if (animTime > animDuration * 0.65f && !activeBoss.hasSlammed) {
                        activeBoss.hasSlammed = true;
                        water.AddRipple(activeBoss.position, 7.0f, 0.35f);
                        if (distToPlayer < 4.0f) {
                            TriggerPlayerDeath();
                        }
                    }

                    if (animTime >= animDuration - 0.05f) {
                        activeBoss.state = 0; // Go back to Idle after slamming
                    }
                }
            }
            
        }

        /// --- DRAW ---
        glm::mat4 proj = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f);       

        // Standard Third-Person View
        float camDist = 1.4f, camHeight = 1.0f, rightOffset = 0.3f;
        glm::vec3 flatFront = glm::normalize(glm::vec3(camera.Front.x, 0.0f, camera.Front.z));
        glm::vec3 flatRight = glm::normalize(glm::cross(flatFront, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 cPos = playerPos - flatFront * camDist + flatRight * rightOffset;
        cPos.y += camHeight;

        if (playerIsDead) cPos.y = glm::max(0.1f, cPos.y - (playerDeathTimer * 0.5f));
        camera.Position = cPos;
        glm::mat4 view = camera.GetViewMatrix();

        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 model = glm::mat4(1.0f);

        glDisable(GL_BLEND);

        staticShader.use();
        staticShader.setMat4("projection", proj);
        staticShader.setMat4("view", view);

        glm::mat4 mapMatrix = glm::mat4(1.0f);
        // You will likely need to adjust this scale/translation to fit your character
        mapMatrix = glm::translate(mapMatrix, glm::vec3(0.0f, -1.0f, 0.0f));
        mapMatrix = glm::scale(mapMatrix, glm::vec3(0.02f)); // Maps are usually massive, scale it down
        staticShader.setMat4("model", mapMatrix);

        mapModel.Draw(staticShader);

        // Water
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(waterProg);
        glUniformMatrix4fv(glGetUniformLocation(waterProg, "projection"), 1, GL_FALSE, &proj[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(waterProg, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(waterProg, "model"), 1, GL_FALSE, &model[0][0]);
        glUniform3fv(glGetUniformLocation(waterProg, "viewPos"), 1, &camera.Position[0]);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glUniform1i(glGetUniformLocation(waterProg, "skybox"), 0);
        water.Draw();


        // Player
        ourShader.use(); ourShader.setMat4("projection", proj); ourShader.setMat4("view", view);
        auto pBone = playerAnimator.GetFinalBoneMatrices();
        for (int i = 0; i < pBone.size(); ++i) ourShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", pBone[i]);
        model = glm::translate(glm::mat4(1.0f), playerPos);
        model = glm::rotate(model, atan2(camera.Front.x, camera.Front.z), glm::vec3(0, 1, 0));
        model = glm::scale(model, glm::vec3(0.5f));
        ourShader.setMat4("model", model);
        ourModel.Draw(ourShader);

        // Boss
        if (!activeBoss.isDead) {
            auto bBones = activeBoss.animator.GetFinalBoneMatrices();
            for (int i = 0; i < bBones.size(); ++i)
                ourShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", bBones[i]);

            model = glm::translate(glm::mat4(1.0f), activeBoss.position);
            glm::vec3 dir = glm::normalize(playerPos - activeBoss.position);
            model = glm::rotate(model, atan2(dir.x, dir.z), glm::vec3(0, 1, 0));

            // Adjust this scale if the Boss imports too small or too large!
            model = glm::scale(model, glm::vec3(1.0f));

            ourShader.setMat4("model", model);
            bossModel.Draw(ourShader);
        }
        fluidSim->simulate(deltaTime, 0.0f, 5); // Using 5 iterations to save CPU time

        // --- 2. MAP TO PIXELS (Corrected Memory Alignment) ---
        for (int j = 0; j < fluidSim->numY; j++) {
            for (int i = 0; i < fluidSim->numX; i++) {
                unsigned char r, g, b;

                // Read from physics engine (Column-Major: i * numY + j)
                float temp = fluidSim->t[i * fluidSim->numY + j];
                getFireColor(temp, r, g, b);

                // Write to GPU texture (Row-Major: j * numX + i)
                int idx = (j * fluidSim->numX + i) * 3;
                firePixelData[idx] = r;
                firePixelData[idx + 1] = g;
                firePixelData[idx + 2] = b;
            }
        }

        // --- 3. UPLOAD TO GPU ---
        glBindTexture(GL_TEXTURE_2D, fireTextureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fluidSim->numX, fluidSim->numY, GL_RGB, GL_UNSIGNED_BYTE, firePixelData.data());

        // --- 4. RENDER BILLBOARDS ---
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);

        glUseProgram(fireProg);

        glUniformMatrix4fv(glGetUniformLocation(fireProg, "projection"), 1, GL_FALSE, &proj[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(fireProg, "view"), 1, GL_FALSE, &view[0][0]);
        glUniform3fv(glGetUniformLocation(fireProg, "cameraRight"), 1, &camera.Right[0]);
        glUniform3fv(glGetUniformLocation(fireProg, "cameraUp"), 1, &camera.Up[0]);

        // ==========================================
        // CHANGED WIDTH FROM 2.0f TO 10.0f
        glm::vec2 fireSize = glm::vec2(10.0f, 6.0f);        
        // ==========================================
        glUniform2fv(glGetUniformLocation(fireProg, "size"), 1, &fireSize[0]);

        glBindVertexArray(fireVAO);
        glBindTexture(GL_TEXTURE_2D, fireTextureID);

        for (const auto& pos : firePositions) {
            glUniform3fv(glGetUniformLocation(fireProg, "centerPos"), 1, &pos[0]);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }

        glDepthMask(GL_TRUE);

        // Skybox
        glDepthFunc(GL_LEQUAL);
        glUseProgram(skyboxProg);
        glm::mat4 skyView = glm::mat4(glm::mat3(camera.GetViewMatrix()));
        glUniformMatrix4fv(glGetUniformLocation(skyboxProg, "view"), 1, GL_FALSE, &skyView[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(skyboxProg, "projection"), 1, GL_FALSE, &proj[0][0]);
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate(); return 0;
}