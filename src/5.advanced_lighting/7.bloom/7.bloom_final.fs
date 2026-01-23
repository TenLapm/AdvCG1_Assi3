#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D scene;      
uniform sampler2D bloomBlur;  
uniform sampler2D depthMap;   

uniform float exposure;

uniform float n     = 0.1;
uniform float f      = 100.0;
uniform float focus_distance = 3.5;  
uniform float focus_range    = 1.0;  

float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; 
    return (2.0 * n * f) / (f + n - z * (f - n));	
}

void main()
{             
    const float gamma = 2.2;

    vec3 sharpColor = texture(scene, TexCoords).rgb;      
    vec3 blurColor  = texture(bloomBlur, TexCoords).rgb;
    
    float depth = LinearizeDepth(texture(depthMap, TexCoords).r);
    float blurAmount = smoothstep(0.0, focus_range, abs(depth - focus_distance));

    vec3 result = mix(sharpColor, blurColor, blurAmount);
    result = vec3(1.0) - exp(-result * exposure);
    result = pow(result, vec3(1.0 / gamma));

    FragColor = vec4(result, 1.0);
}