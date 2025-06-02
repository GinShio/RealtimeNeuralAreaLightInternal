import "../rtxns/Activation";
import "../rtxns/CooperativeVectorAutoDiff";
import "../rtxns/CooperativeVectorDerivatives";
import "../rtxns/CooperativeVectorFunctions";
import "../rtxns/LinearOps";
import "../rtxns/Loss";
import "../rtxns/MLP";
import "../rtxns/PCG32";
import "../rtxns/Utils";

import "disney";

struct PushConstants
{
    uint64_t seed;
}

[[push_constant]]
PushConstants pc;

struct UniformBuffer
{
    uint dataSize;
    uint textureSize;
    uint pixelCount;
}

[[vk::binding(0)]]
uniform ConstantBuffer<UniformBuffer> uniforms;

[[vk::binding(1)]]
RWStructuredBuffer<half> data;

[[vk::binding(2)]]
RWStructuredBuffer<half> materialData;

// rotation from normal to (0, 0, 1)
func tangentFrame(normal: float3)->float3x3
{
    let up = abs(normal.z) < 0.9999 ? float3(0, 0, 1) : float3(1, 0, 0);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return float3x3(tangent, bitangent, normal);
}

func sampleWiWo(
    inout rng: PCG32,
    out wi: float3,
    out wo: float3)
{
    for (var i = 0; i < 10; ++i)
    {
        let zH = rng.nextFloat();
        let rH = sqrt(1.0 - zH * zH);
        let phiH = 2.0 * PI * rng.nextFloat();
        let H = float3(
            rH * cos(phiH),
            rH * sin(phiH),
            zH);

        let zDelta = 2.0 * rng.nextFloat() - 1.0;
        let rDelta = sqrt(1.0 - zDelta * zDelta);
        let phiDelta = 2.0 * PI * rng.nextFloat();
        let delta = float3(
            rDelta * cos(phiDelta),
            rDelta * sin(phiDelta),
            zDelta);

        let M = transpose(tangentFrame(H));
        let d = mul(M, delta);

        wi = normalize(d);
        wo = normalize(2.0 * dot(wi, H) * H - wi);

        if (wo.z >= 0.0 && wi.z >= 0.0)
        {
            return;
        }
    }
}

func actualBrdfFunc(
    inout rng: PCG32,
    wi: float3,
    wo: float3,
    baseColor: float3,
    roughness: float,
    metallic: float,
    normal: float3)
    ->float3
{
    let N = normalize(normal * 2 - 1);

    let TBN = tangentFrame(N);

    let wi_tangent = mul(TBN, wi);
    let wo_tangent = mul(TBN, wo);
    return DisneyBRDF(wi_tangent, wo_tangent, baseColor.rgb, roughness, metallic);
}

func log1p(float x)->float
{
    return log(x + 1.0);
}

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= uniforms.dataSize)
        return;

    PCG32 rng = PCG32(pc.seed, i);

    // sample wi and wo
    float3 wi;
    float3 wo;
    sampleWiWo(rng, wi, wo);

    // sample parameters
    let materialBaseIndex = (i % uniforms.pixelCount) * 8;
    let baseColor = float3(
        materialData[materialBaseIndex + 0],
        materialData[materialBaseIndex + 1],
        materialData[materialBaseIndex + 2]);
    let roughness = materialData[materialBaseIndex + 3];
    let metallic = materialData[materialBaseIndex + 4];
    let normal = float3(
        materialData[materialBaseIndex + 5],
        materialData[materialBaseIndex + 6],
        materialData[materialBaseIndex + 7]);

    // compute brdf
    let brdf = actualBrdfFunc(
        rng,
        wi,
        wo,
        baseColor,
        roughness,
        metallic,
        normal);

    // Save training data
    let index = i * 9;
    data[index + 0] = half(wi.x);
    data[index + 1] = half(wi.y);
    data[index + 2] = half(wi.z);
    data[index + 3] = half(wo.x);
    data[index + 4] = half(wo.y);
    data[index + 5] = half(wo.z);
    data[index + 6] = half(log1p(brdf.x));
    data[index + 7] = half(log1p(brdf.y));
    data[index + 8] = half(log1p(brdf.z));
}
