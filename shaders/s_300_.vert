#version 450
#extension GL_ARB_separate_shader_objects : enable



layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;


// layout(push_constant) uniform PushConstants {
//     mat4 view;
// } pcs;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;


layout(location = 0) out vec3 fragColor;


mat4 t1 = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 2.0
);

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition);
    fragColor = vec3(inColor);
}
