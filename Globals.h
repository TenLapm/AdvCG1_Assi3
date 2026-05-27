#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>

#include <learnopengl/camera.h>
#include <learnopengl/animator.h>
#include <learnopengl/model_animation.h>

// --- SCREEN & CAMERA ---
extern const unsigned int SCR_WIDTH;
extern const unsigned int SCR_HEIGHT;
extern Camera camera;
extern float lastX;
extern float lastY;
extern bool firstMouse;

// --- TIME ---
extern float deltaTime;
extern float lastFrame;

// --- PLAYER STATE ---
extern glm::vec3 playerPos;
extern float playerSpeed;
extern float playerRadius;
extern bool playerIsDead;
extern float playerDeathTimer;

// --- ANIMATIONS ---
extern Animation* pIdleAnim;
extern Animation* pWalkW;
extern Animation* pWalkS;
extern Animation* pStrafeA;
extern Animation* pStrafeD;
extern Animation* pDeathAnim;
extern Animation* pRunAnim;
extern Animation* pPlayerSlash;   
extern Animation* pBossJumpAttack; 
extern Animation* currentMovementAnim;

// --- GAME STATE & WAVES ---
extern bool gameStarted;
extern int currentWave;
extern bool playerIsAttacking;
extern int enemiesSpawnedInWave;
extern int enemiesToSpawnTotal;

// --- SETTINGS ---
extern float ENEMY_SPAWN_RADIUS_MIN;
extern float ENEMY_SPAWN_RADIUS_MAX;
extern float ENEMY_SPEED;
extern float ENEMY_HP;
extern float ENEMY_HITBOX_RADIUS;

// --- DATA STRUCTURES ---
struct Obstacle {
    glm::vec3 position;
    glm::vec3 size;
};
extern std::vector<Obstacle> mapObstacles;


struct Boss {
    glm::vec3 position;
    float hp;
    float speed;
    int state; // 0 = Chasing, 1 = Attacking
    bool hasSlammed;
    Animator animator;
    bool isDead;

    Boss() : animator(nullptr), hp(50.0f), speed(2.5f), state(0), hasSlammed(false), isDead(false) {}
};
extern Boss activeBoss;