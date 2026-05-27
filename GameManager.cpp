#include "GameManager.h"
#include "Physics.h"
#include <iostream>
#include <cstdlib>

void CreateMap() {
    mapObstacles.clear();
}

void ManageWaves(float dt) {
    
}

void SpawnEnemyInFOV() {
    
}

void TriggerPlayerDeath() {
    if (playerIsDead) return;
    std::cout << "PLAYER DIED!" << std::endl;
    playerIsDead = true;
    playerDeathTimer = 0.0f;
}

void ResetGame() {
    glm::vec3 playerPos(-22.382f, -0.4f, 11.1701f);
    gameStarted = false;
    playerIsDead = false;
    playerIsAttacking = false;
    playerDeathTimer = 0.0f;

    // Reset the Boss: Spawn at x -5, z 5
    activeBoss.position = glm::vec3(-5.0f, -0.4f, 5.0f);
    activeBoss.hp = 50.0f;
    activeBoss.state = 0;
    activeBoss.isDead = false;
    activeBoss.hasSlammed = false;

    glfwSetInputMode(glfwGetCurrentContext(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    std::cout << "RESET GAME - Click to Retry" << std::endl;
}