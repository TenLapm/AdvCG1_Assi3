#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 envRotation;

out vec3 WorldPos;

void main()
{
    // Rotate the skybox geometry
    WorldPos = vec3(envRotation * vec4(aPos, 1.0));

    mat4 rotView = mat4(mat3(view));
    vec4 clipPos = projection * rotView * vec4(WorldPos, 1.0);

    gl_Position = clipPos.xyww;
}