#pragma once
#include <glm/glm.hpp>
#include "Globals.h"

bool CheckWallCollision(glm::vec3 position, float radius, const Obstacle& obs);
bool IsCollidingWithMap(glm::vec3 position, float radius);
bool IsInsideFOV(glm::vec3 targetPos, const Camera& cam, float fovDeg);
glm::vec3 MoveWithSliding(glm::vec3 currentPos, glm::vec3 velocity, float radius);
glm::vec3 GetSmartPath(glm::vec3 startPos, glm::vec3 targetPos);
float GetRayAABBIntersection(glm::vec3 rayOrigin, glm::vec3 rayDir, const Obstacle& obs);
float GetRaySphereIntersection(glm::vec3 rayOrigin, glm::vec3 rayDir, glm::vec3 sphereCenter, float sphereRadius);