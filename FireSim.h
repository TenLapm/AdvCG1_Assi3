#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

inline float randomFloat() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }

const int U_FIELD = 0;
const int V_FIELD = 1;
const int T_FIELD = 2;

void getFireColor(float val, unsigned char& r, unsigned char& g, unsigned char& b) {
    val = std::min(std::max(val, 0.0f), 1.0f);
    float fr, fg, fb;
    if (val < 0.3f) {
        float s = val / 0.3f;
        fr = 0.2f * s; fg = 0.2f * s; fb = 0.2f * s;
    }
    else if (val < 0.5f) {
        float s = (val - 0.3f) / 0.2f;
        fr = 0.2f + 0.8f * s; fg = 0.1f; fb = 0.1f;
    }
    else {
        float s = (val - 0.5f) / 0.48f;
        fr = 1.0f; fg = s; fb = 0.0f;
    }
    r = static_cast<unsigned char>(255.0f * fr);
    g = static_cast<unsigned char>(255.0f * fg);
    b = static_cast<unsigned char>(255.0f * fb);
}

class Fluid {
public:
    int numX, numY, numCells;
    float h;
    std::vector<float> u, v, newU, newV, s, t, newT;
    int numSwirls = 0;
    int maxSwirls = 100;
    std::vector<float> swirlX, swirlY, swirlOmega, swirlRadius, swirlTime;

    // Settings moved inside class for easy access
    bool burningObstacle = true;
    bool burningFloor = false;
    float obstacleX = 0.0f;
    float obstacleY = 0.0f;
    float obstacleRadius = 0.2f;
    float swirlProbability = 80.0f;
    float swirlMaxRadius = 0.04f;

    Fluid(int numX_in, int numY_in, float h_in) {
        numX = numX_in + 2; numY = numY_in + 2;
        numCells = numX * numY; h = h_in;
        u.resize(numCells, 0.0f); v.resize(numCells, 0.0f);
        newU.resize(numCells, 0.0f); newV.resize(numCells, 0.0f);
        s.resize(numCells, 1.0f); t.resize(numCells, 0.0f); newT.resize(numCells, 0.0f);
        swirlX.resize(maxSwirls, 0.0f); swirlY.resize(maxSwirls, 0.0f);
        swirlOmega.resize(maxSwirls, 0.0f); swirlRadius.resize(maxSwirls, 0.0f);
        swirlTime.resize(maxSwirls, 0.0f);
    }

    void integrate(float dt, float gravity) {
        int n = numY;
        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[i * n + j] != 0.0f && s[i * n + j - 1] != 0.0f) v[i * n + j] += gravity * dt;
            }
        }
    }

    void solveIncompressibility(int numIters, float dt) {
        int n = numY; float overRelaxation = 1.9f;
        for (int iter = 0; iter < numIters; iter++) {
            for (int i = 1; i < numX - 1; i++) {
                for (int j = 1; j < numY - 1; j++) {
                    if (s[i * n + j] == 0.0f) continue;
                    float sx0 = s[(i - 1) * n + j], sx1 = s[(i + 1) * n + j];
                    float sy0 = s[i * n + j - 1], sy1 = s[i * n + j + 1];
                    float sumS = sx0 + sx1 + sy0 + sy1;
                    if (sumS == 0.0f) continue;
                    float div = u[(i + 1) * n + j] - u[i * n + j] + v[i * n + j + 1] - v[i * n + j];
                    float p = -div / sumS; p *= overRelaxation;
                    u[i * n + j] -= sx0 * p; u[(i + 1) * n + j] += sx1 * p;
                    v[i * n + j] -= sy0 * p; v[i * n + j + 1] += sy1 * p;
                }
            }
        }
    }

    void extrapolate() {
        int n = numY;
        for (int i = 0; i < numX; i++) { u[i * n + 0] = u[i * n + 1]; u[i * n + numY - 1] = u[i * n + numY - 2]; }
        for (int j = 0; j < numY; j++) { v[0 * n + j] = v[1 * n + j]; v[(numX - 1) * n + j] = v[(numX - 2) * n + j]; }
    }

    float sampleField(float x, float y, int field) {
        int n = numY; float h1 = 1.0f / h, h2 = 0.5f * h;
        x = std::max(std::min(x, numX * h), h); y = std::max(std::min(y, numY * h), h);
        float dx = 0.0f, dy = 0.0f; const std::vector<float>* f = nullptr;
        switch (field) { case U_FIELD: f = &u; dy = h2; break; case V_FIELD: f = &v; dx = h2; break; case T_FIELD: f = &t; dx = h2; dy = h2; break; }
                                     int x0 = std::min(static_cast<int>(std::floor((x - dx) * h1)), numX - 1); float tx = ((x - dx) - x0 * h) * h1; int x1 = std::min(x0 + 1, numX - 1);
                                     int y0 = std::min(static_cast<int>(std::floor((y - dy) * h1)), numY - 1); float ty = ((y - dy) - y0 * h) * h1; int y1 = std::min(y0 + 1, numY - 1);
                                     float sx = 1.0f - tx, sy = 1.0f - ty;
                                     return sx * sy * (*f)[x0 * n + y0] + tx * sy * (*f)[x1 * n + y0] + tx * ty * (*f)[x1 * n + y1] + sx * ty * (*f)[x0 * n + y1];
    }

    float avgU(int i, int j) { return (u[i * numY + j - 1] + u[i * numY + j] + u[(i + 1) * numY + j - 1] + u[(i + 1) * numY + j]) * 0.25f; }
    float avgV(int i, int j) { return (v[(i - 1) * numY + j] + v[i * numY + j] + v[(i - 1) * numY + j + 1] + v[i * numY + j + 1]) * 0.25f; }

    void advectVel(float dt) {
        newU = u; newV = v; int n = numY; float h2 = 0.5f * h;
        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY; j++) {
                if (s[i * n + j] != 0.0f && s[(i - 1) * n + j] != 0.0f && j < numY - 1) {
                    float x = i * h, y = j * h + h2; float velU = u[i * n + j], velV = avgV(i, j);
                    x = x - dt * velU; y = y - dt * velV; newU[i * n + j] = sampleField(x, y, U_FIELD);
                }
                if (s[i * n + j] != 0.0f && s[i * n + j - 1] != 0.0f && i < numX - 1) {
                    float x = i * h + h2, y = j * h; float velU = avgU(i, j), velV = v[i * n + j];
                    x = x - dt * velU; y = y - dt * velV; newV[i * n + j] = sampleField(x, y, V_FIELD);
                }
            }
        }
        u = newU; v = newV;
    }

    void advectTemperature(float dt) {
        newT = t; int n = numY; float h2 = 0.5f * h;
        for (int i = 1; i < numX - 1; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[i * n + j] != 0.0f) {
                    float velU = (u[i * n + j] + u[(i + 1) * n + j]) * 0.5f; float velV = (v[i * n + j] + v[i * n + j + 1]) * 0.5f;
                    float x = i * h + h2 - dt * velU; float y = j * h + h2 - dt * velV;
                    newT[i * n + j] = sampleField(x, y, T_FIELD);
                }
            }
        }
        t = newT;
    }

    void updateFire(float dt) {
        float swirlTimeSpan = 1.0f; 

        // Extreme turbulence to whip the flames around
        float swirlOmegaForce = 80.0f;

        float swirlDamping = 10.0f * dt;
        float swirlProb = swirlProbability * h * h;

        // EXTREMELY low cooling so the fire survives all the way to the top
        float fireCooling = 0.1f * dt;

        float smokeCooling = 0.3f * dt;

        // MASSIVE lift and acceleration to shoot the fluid to the top of the grid
        float lift = 25.0f;
        float acceleration = 30.0f * dt;

        float kernelRadius = swirlMaxRadius;
        int n = numY; float maxX = (numX - 1) * h, maxY = (numY - 1) * h;
        // ==========================================

        int num = 0;
        for (int nr = 0; nr < numSwirls; nr++) {
            swirlTime[nr] -= dt;
            if (swirlTime[nr] > 0.0f) {
                swirlTime[num] = swirlTime[nr]; swirlX[num] = swirlX[nr]; swirlY[num] = swirlY[nr]; swirlOmega[num] = swirlOmega[nr]; num++;
            }
        }
        numSwirls = num;

        for (int nr = 0; nr < numSwirls; nr++) {
            float x = swirlX[nr], y = swirlY[nr];
            float swirlU = (1.0f - swirlDamping) * sampleField(x, y, U_FIELD);
            float swirlV = (1.0f - swirlDamping) * sampleField(x, y, V_FIELD);
            x += swirlU * dt; y += swirlV * dt;
            x = std::min(std::max(x, h), maxX); y = std::min(std::max(y, h), maxY);
            swirlX[nr] = x; swirlY[nr] = y;
            float omega = swirlOmega[nr];

            int x0 = std::max(static_cast<int>(std::floor((x - kernelRadius) / h)), 0);
            int y0 = std::max(static_cast<int>(std::floor((y - kernelRadius) / h)), 0);
            int x1 = std::min(static_cast<int>(std::floor((x + kernelRadius) / h)) + 1, numX - 1);
            int y1 = std::min(static_cast<int>(std::floor((y + kernelRadius) / h)) + 1, numY - 1);

            for (int i = x0; i <= x1; i++) {
                for (int j = y0; j <= y1; j++) {
                    for (int dim = 0; dim < 2; dim++) {
                        float vx = (dim == 0) ? i * h : (i + 0.5f) * h; float vy = (dim == 0) ? (j + 0.5f) * h : j * h;
                        float rx = vx - x, ry = vy - y; float r = std::sqrt(rx * rx + ry * ry);
                        if (r < kernelRadius) {
                            float strength = 1.0f; if (r > 0.8f * kernelRadius) strength = 5.0f - 5.0f / kernelRadius * r;
                            if (dim == 0) { float target = ry * omega + swirlU; float velU = u[n * i + j]; u[n * i + j] = velU + (target - velU) * strength; }
                            else { float target = -rx * omega + swirlV; float velV = v[n * i + j]; v[n * i + j] += (target - velV) * strength; }
                        }
                    }
                }
            }
        }

        // --- OLD minR / maxR VARIABLES DELETED ---

        for (int i = 0; i < numX; i++) {
            for (int j = 0; j < numY; j++) {
                float temp = t[i * n + j]; float cooling = temp < 0.3f ? smokeCooling : fireCooling;
                t[i * n + j] = std::max(temp - cooling, 0.0f);
                float velV = v[i * n + j], targetV = t[i * n + j] * lift;
                v[i * n + j] += (targetV - velV) * acceleration;
                int numNewSwirls = 0;

                // --- NEW STRAIGHT LINE LOGIC ---
                if (burningObstacle) {
                    float dx = std::abs((i + 0.5f) * h - obstacleX);
                    float dy = std::abs((j + 0.5f) * h - obstacleY - 1.0f * h); // Lowered the spawn point slightly

                    float lineHalfLength = obstacleRadius * 3.0f; // Controls how WIDE the line of fire is
                    float lineThickness = h * 2.0f;               // Controls how THICK the base of the fire is

                    // Uses a rectangular bounding box instead of radius!
                    if (dx < lineHalfLength && dy < lineThickness) {
                        t[i * n + j] = 1.0f;
                        if (randomFloat() < 0.5f * swirlProb) numNewSwirls++;
                    }
                }

                if (j < 4 && burningFloor) {
                    t[i * n + j] = 1.0f; u[i * n + j] = 0.0f; v[i * n + j] = 0.0f;
                    if (randomFloat() < swirlProb) numNewSwirls++;
                }
                for (int k = 0; k < numNewSwirls; k++) {
                    if (numSwirls >= maxSwirls) break;
                    int nr = numSwirls; swirlX[nr] = i * h; swirlY[nr] = j * h;
                    swirlOmega[nr] = (-1.0f + 2.0f * randomFloat()) * swirlOmegaForce; swirlTime[nr] = swirlTimeSpan;
                    numSwirls++;
                }
            }
        }

        for (int i = 1; i < numX - 1; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (t[i * n + j] == 1.0f) {
                    t[i * n + j] = (t[(i - 1) * n + (j - 1)] + t[(i + 1) * n + (j - 1)] + t[(i + 1) * n + (j + 1)] + t[(i - 1) * n + (j + 1)]) * 0.25f;
                }
            }
        }
    }
    void simulate(float dt, float gravity, int numIters) {
        integrate(dt, gravity);
        solveIncompressibility(numIters, dt);
        extrapolate();
        advectVel(dt);
        advectTemperature(dt);
        updateFire(dt);
    }
};