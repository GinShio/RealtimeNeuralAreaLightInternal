import "../base.slang";

import "../../rtxns/Activation";
import "../../rtxns/CooperativeVectorAutoDiff";
import "../../rtxns/CooperativeVectorDerivatives";
import "../../rtxns/CooperativeVectorFunctions";
import "../../rtxns/LinearOps";
import "../../rtxns/Loss";
import "../../rtxns/MLP";
import "../../rtxns/PCG32";
import "../../rtxns/Utils";

import "network";
import "modify-vertices";

using namespace rtxns;
using namespace rtxns::mlp;

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
    float4x4 model;
    uint4 weightOffsets0;
    uint4 weightOffsets1;
    uint4 biasOffsets0;
    uint4 biasOffsets1;
    uint latentTexture0Index;
    uint latentTexture1Index;
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
    float3 viewPosition : TEXCOORD0;
    float3 viewNormal : TEXCOORD1;
    float4 viewTangent : TEXCOORD2;
    float2 uv : TEXCOORD3;
}

float3x3 inverse3x3(float3x3 m)
{
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

    float det = a00 * (a11 * a22 - a12 * a21) -
                a01 * (a10 * a22 - a12 * a20) +
                a02 * (a10 * a21 - a11 * a20);

    if (abs(det) < 1e-6)
        return float3x3(0.0);

    float invDet = 1.0 / det;

    return float3x3(
        +(a11 * a22 - a12 * a21) * invDet,
        -(a01 * a22 - a02 * a21) * invDet,
        +(a01 * a12 - a02 * a11) * invDet,

        -(a10 * a22 - a12 * a20) * invDet,
        +(a00 * a22 - a02 * a20) * invDet,
        -(a00 * a12 - a02 * a10) * invDet,

        +(a10 * a21 - a11 * a20) * invDet,
        -(a00 * a21 - a01 * a20) * invDet,
        +(a00 * a11 - a01 * a10) * invDet);
}

func srgbToLinear(color: float3)->float3
{
    let useLinear = color <= 0.04045;
    let v1 = color / 12.92;
    let v2 = pow((color + 0.055) / 1.055, 2.4);
    return select(useLinear, v1, v2);
}

[shader("vertex")]
func vsMain(input: VertexInput)->VertexOutput
{
    let viewModel = mul(uniforms.view, pc.model);
    let viewModelIT = inverse3x3(transpose(float3x3(viewModel)));

    let viewPosition = mul(viewModel, float4(input.position, 1.0));

    let viewNormal = normalize(mul(viewModelIT, input.normal));

    let viewTangent = normalize(mul(viewModelIT, input.tangent.xyz));
    let viewTangentW = input.tangent.w;

    let uv = input.uv;

    let position = mul(uniforms.projection, viewPosition);

    return {
        position,
        viewPosition.xyz,
        viewNormal.xyz,
        float4(viewTangent.xyz, viewTangentW),
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
    let sampler = samplerList[uniforms.samplerIndex];

    let latentValue0 = textures[pc.latentTexture0Index].Sample(sampler, input.uv);
    let latentValue1 = textures[pc.latentTexture1Index].Sample(sampler, input.uv);
    CoopVec<half, 8> latentVector = {
        latentValue0.r,
        latentValue0.g,
        latentValue0.b,
        latentValue0.a,
        latentValue1.r,
        latentValue1.g,
        latentValue1.b,
        latentValue1.a,
    };

    var network = Network(
        params,
        pc.weightOffsets0,
        pc.weightOffsets1,
        pc.biasOffsets0,
        pc.biasOffsets1);

    let normal = normalize(input.viewNormal);
    let tangent = normalize(input.viewTangent.xyz);

    let N = normalize(normal);
    let T = normalize(tangent - dot(tangent, N) * N);
    let B = cross(N, T) * input.viewTangent.w;
    let TBN = float3x3(T, B, N);

    let Vt = mul(TBN, normalize(-input.viewPosition));
    let Nt = float3(0, 0, 1);

    var color = float3(0.0, 0.0, 0.0);
    for (var i = 0; i < uniforms.lightCount; i++)
    {
        let light = uniforms.light[i];

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
        v1 = mul(light_rot_scale, v1) + light.position;
        v2 = mul(light_rot_scale, v2) + light.position;
        v3 = mul(light_rot_scale, v3) + light.position;
        v4 = mul(light_rot_scale, v4) + light.position;

        v1 = mul(uniforms.view, float4(v1, 1.0)).xyz;
        v2 = mul(uniforms.view, float4(v2, 1.0)).xyz;
        v3 = mul(uniforms.view, float4(v3, 1.0)).xyz;
        v4 = mul(uniforms.view, float4(v4, 1.0)).xyz;

        v1 -= input.viewPosition;
        v2 -= input.viewPosition;
        v3 -= input.viewPosition;
        v4 -= input.viewPosition;

        v1 = normalize(mul(TBN, v1));
        v2 = normalize(mul(TBN, v2));
        v3 = normalize(mul(TBN, v3));
        v4 = normalize(mul(TBN, v4));

        modifyVertices(Vt, v1, v2, v3, v4);

        if (v1.z < 0.0 && v2.z < 0.0 && v3.z < 0.0 && v4.z < 0.0)
        {
            continue;
        }

        if (dot(Vt, Nt) < 0.0)
        {
            continue;
        }

        let output = network.forward(
            latentVector,
            normalize(Vt),
            normalize(v1), normalize(v2), normalize(v3), normalize(v4));

        let D = float3(output[0], output[1], output[2]);
        let lightColor = srgbToLinear(light.color);
        color += D * lightColor * light.intensity;
    }

    return {
        { color, 1.0 },
    };
}
