import "../base.slang";

struct LightParams
{
    float3 position;
    float intensity;
    float3 rotation;
    float size;
    float3 color;
    float aspectRatio;
}

struct Uniforms
{
    float4x4 view;
    float4x4 projection;

    uint samplerIndex;

    uint lightCount;

    LightParams light[3];
}

struct PushConstants
{
    uint lightIndex;
}

[[push_constant]]
PushConstants pc;

[[vk::binding(0, 2)]]
uniform ConstantBuffer<Uniforms> uniforms;

[[vk::binding(0, 3)]]
StructuredBuffer<half> params;

struct VertexOutput
{
    float4 position : SV_POSITION;
    float3 _normal : NORMAL;
    float4 _tangent : TANGENT;
    float2 uv : TEXCOORD3;
}

[shader("vertex")]
func vsMain(input: VertexInput)->VertexOutput
{
    let light = uniforms.light[pc.lightIndex];

    var v1 = float3(0.5, 0.5, 0.0);
    var v2 = float3(0.5, -0.5, 0.0);
    var v3 = float3(-0.5, -0.5, 0.0);
    var v4 = float3(-0.5, 0.5, 0.0);

    let light_size = float3x3(
        light.size, 0.0, 0.0,
        0.0, light.size, 0.0,
        0.0, 0.0, light.size);
    let light_aspect = float3x3(
        light.aspectRatio, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);
    let light_scale = mul(light_size, light_aspect);
    let light_rotation_z = float3x3(
        cos(light.rotation.z), -sin(light.rotation.z), 0.0,
        sin(light.rotation.z), cos(light.rotation.z), 0.0,
        0.0, 0.0, 1.0);
    let light_rotation_y = float3x3(
        cos(light.rotation.y), 0.0, sin(light.rotation.y),
        0.0, 1.0, 0.0,
        -sin(light.rotation.y), 0.0, cos(light.rotation.y));
    let light_rotation_x = float3x3(
        1.0, 0.0, 0.0,
        0.0, cos(light.rotation.x), -sin(light.rotation.x),
        0.0, sin(light.rotation.x), cos(light.rotation.x));
    let light_rotation = mul(mul(light_rotation_y, light_rotation_x), light_rotation_z);

    let light_rot_scale = mul(light_rotation, light_scale);

    let model = float4x4(
        light_rot_scale[0][0], light_rot_scale[0][1], light_rot_scale[0][2], light.position.x,
        light_rot_scale[1][0], light_rot_scale[1][1], light_rot_scale[1][2], light.position.y,
        light_rot_scale[2][0], light_rot_scale[2][1], light_rot_scale[2][2], light.position.z,
        0.0, 0.0, 0.0, 1.0);

    let viewModel = mul(uniforms.view, model);
    let viewPosition = mul(viewModel, float4(input.position, 1.0));
    let position = mul(uniforms.projection, viewPosition);

    let uv = input.uv;

    return {
        position,
        input.normal,
        input.tangent,
        uv,
    };
}

struct FragmentOutput
{
    float4 color : SV_TARGET;
}

[shader("fragment")]
func fsMain(input: VertexOutput)->FragmentOutput
{
    let light = uniforms.light[pc.lightIndex];
    let color = light.color * light.intensity;
    return {
        { color, 1.0 },
    };
}
