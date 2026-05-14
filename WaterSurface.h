#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

struct WaterVertex {
    glm::vec3 Position;
    glm::vec3 Normal;
};

class WaterSurface {
public:
    int numX, numZ;
    float spacing;
    float baseDepth;

    // Physics parameters
    float waveSpeed = 6.0f;
    float damping = 2.5f;   // Friction to slow waves down
    float spring = 15.0f;   // How fast it tries to return to being flat

    std::vector<WaterVertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<float> heights;
    std::vector<float> velocities;

    unsigned int VAO, VBO, EBO;

    WaterSurface(float sizeX, float sizeZ, float depth, float gridSpacing) {
        spacing = gridSpacing;
        baseDepth = depth;
        numX = std::floor(sizeX / spacing) + 1;
        numZ = std::floor(sizeZ / spacing) + 1;
        int numCells = numX * numZ;

        heights.resize(numCells, depth);
        velocities.resize(numCells, 0.0f);
        vertices.resize(numCells);

        int cx = numX / 2;
        int cz = numZ / 2;

        for (int i = 0; i < numX; i++) {
            for (int j = 0; j < numZ; j++) {
                int id = i * numZ + j;
                vertices[id].Position = glm::vec3((i - cx) * spacing, depth, (j - cz) * spacing);
                vertices[id].Normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }
        }

        for (int i = 0; i < numX - 1; i++) {
            for (int j = 0; j < numZ - 1; j++) {
                int id0 = i * numZ + j;
                int id1 = i * numZ + j + 1;
                int id2 = (i + 1) * numZ + j + 1;
                int id3 = (i + 1) * numZ + j;
                indices.push_back(id0); indices.push_back(id1); indices.push_back(id2);
                indices.push_back(id0); indices.push_back(id2); indices.push_back(id3);
            }
        }

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(WaterVertex), vertices.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(WaterVertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(WaterVertex), (void*)offsetof(WaterVertex, Normal));
        glBindVertexArray(0);
    }

    void AddRipple(glm::vec3 worldPos, float radius, float strength) {
        int cx = numX / 2;
        int cz = numZ / 2;
        int gridX = cx + std::round(worldPos.x / spacing);
        int gridZ = cz + std::round(worldPos.z / spacing);
        int r = std::max(1, (int)(radius / spacing));

        float r2 = radius * radius;
        for (int i = -r; i <= r; i++) {
            for (int j = -r; j <= r; j++) {
                int nx = gridX + i;
                int nz = gridZ + j;
                // Keep away from the extreme edges to prevent errors
                if (nx > 1 && nx < numX - 2 && nz > 1 && nz < numZ - 2) {
                    float dist2 = (i * spacing) * (i * spacing) + (j * spacing) * (j * spacing);
                    if (dist2 < r2) {
                        // Smooth bell curve falloff for beautiful ripples
                        float falloff = 1.0f - (dist2 / r2);
                        heights[nx * numZ + nz] -= strength * falloff;
                    }
                }
            }
        }
    }

    void Simulate(float dt) {
        if (dt <= 0.0001f) return;

        // 1. Calculate wave acceleration
        for (int i = 1; i < numX - 1; i++) {
            for (int j = 1; j < numZ - 1; j++) {
                int id = i * numZ + j;
                float h = heights[id];

                // Average of 4 neighbors
                float sumH = heights[id - 1] + heights[id + 1] + heights[id - numZ] + heights[id + numZ];

                // Laplacian equation for wave travel
                float accel = (sumH - 4.0f * h) * (waveSpeed * waveSpeed);
                // Spring force to return to flat water
                accel += (baseDepth - h) * spring;

                velocities[id] += accel * dt;
                velocities[id] -= velocities[id] * damping * dt; // Apply friction
            }
        }

        // 2. Apply velocities to height
        for (int i = 0; i < numX * numZ; i++) {
            heights[i] += velocities[i] * dt;
            vertices[i].Position.y = heights[i];
        }

        ComputeNormals();

        // 3. Send to GPU
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(WaterVertex), vertices.data());
    }

    void ComputeNormals() {
        for (int i = 0; i < numX * numZ; i++) vertices[i].Normal = glm::vec3(0.0f);

        for (size_t i = 0; i < indices.size(); i += 3) {
            int i1 = indices[i];
            int i2 = indices[i + 1];
            int i3 = indices[i + 2];

            glm::vec3 v1 = vertices[i1].Position;
            glm::vec3 v2 = vertices[i2].Position;
            glm::vec3 v3 = vertices[i3].Position;

            glm::vec3 normal = glm::cross(v2 - v1, v3 - v1);
            vertices[i1].Normal += normal;
            vertices[i2].Normal += normal;
            vertices[i3].Normal += normal;
        }

        for (int i = 0; i < numX * numZ; i++) {
            vertices[i].Normal = glm::normalize(vertices[i].Normal);
        }
    }

    void Draw() {
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};