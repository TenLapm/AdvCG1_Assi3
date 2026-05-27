#define NOMINMAX
#include "Physics.h"
#include <algorithm>

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