#include "Globals.h"
#include "Input.h"
#include "Physics.h"

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    glm::vec3 f = glm::normalize(glm::vec3(camera.Front.x, 0, camera.Front.z));
    glm::vec3 r = glm::normalize(glm::cross(f, glm::vec3(0, 1, 0)));
    glm::vec3 n = playerPos;

    // Default to idle if no keys are pressed
    currentMovementAnim = pIdleAnim;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        n += f * playerSpeed * deltaTime;
        currentMovementAnim = pWalkW;
    }
    else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        n -= f * playerSpeed * deltaTime;
        currentMovementAnim = pWalkS;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        n -= r * playerSpeed * deltaTime;
        currentMovementAnim = pStrafeA;
    }
    else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        n += r * playerSpeed * deltaTime;
        currentMovementAnim = pStrafeD;
    }

    if (!IsCollidingWithMap(n, playerRadius)) playerPos = n;
}

void framebuffer_size_callback(GLFWwindow* w, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* w, double x, double y) {
    if (!gameStarted || playerIsDead) return;
    if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }

    float xo = x - lastX;
    float yo = lastY - y;

    camera.ProcessMouseMovement(xo, yo);
    lastX = x;
    lastY = y;
}

void scroll_callback(GLFWwindow* w, double x, double y) {
    camera.ProcessMouseScroll(y);
}