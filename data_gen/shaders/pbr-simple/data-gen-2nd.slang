import "../rtxns/Activation";
import "../rtxns/CooperativeVectorAutoDiff";
import "../rtxns/CooperativeVectorDerivatives";
import "../rtxns/CooperativeVectorFunctions";
import "../rtxns/LinearOps";
import "../rtxns/Loss";
import "../rtxns/MLP";
import "../rtxns/PCG32";
import "../rtxns/Utils";

import "sample-direction";
import "clipping";
import "brdf";

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
    float maxLightSize;
    float maxLightDistance;
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

        if (wo.z >= 0.0)
        {
            return;
        }
    }
}

func computeDistribution(
    inout rng: PCG32,
    baseColor: float3,
    roughness: float,
    metallic: float,
    normal: float3,
    wo: float3,
    v1: float3,
    v2: float3,
    v3: float3,
    v4: float3)
    ->float3
{
    const uint ITERATIONS = 1024;

    float3 D = float3(0.0, 0.0, 0.0);

    let wt = WeightedTriangles.clipAndTriangulateQuad(v1, v2, v3, v4);

    if (wt.triCount == 0)
    {
        return D; // No triangles to sample from
    }

    for (uint i = 0; i < ITERATIONS; ++i)
    {

        let u = rng.nextFloat();
        let v = rng.nextFloat();

        let triangle = wt.sample(rng);
        let bary = triangle.randomBarycentric(rng);
        let lightPos = triangle.barycentricToPosition(bary);
        let lightNormal = triangle.normal();

        let wi = normalize(lightPos);
        let NdotL = saturate(dot(normal, wi));
        let LnDotWi = saturate(abs(dot(lightNormal, -wi)));

        let distanceSquared = dot(lightPos, lightPos);
        let G = (NdotL * LnDotWi) / (distanceSquared + 1e-5);

        if (G > 0.0)
        {
            let brdf = PbrBRDF(baseColor, roughness, metallic, normal, wo, wi);
            D += brdf * G * wt.totalArea;
        }
    }

    return D / float(ITERATIONS);
}

func log1p(x: float)->float
{
    var ret = x;
    ret = log(1.0 + ret);
    return ret;
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

    float3 v1;
    float3 v2;
    float3 v3;
    float3 v4;
    sampleDirection(rng, wi, v1, v2, v3, v4);

    // compute D
    float3 D;
    if (rng.nextFloat() < 0.5)
    {
        D = computeDistribution(
            rng,
            baseColor,
            roughness,
            metallic,
            normal,
            wo,
            v1, v2, v3, v4);
    }
    else
    {

        D = computeDistribution(
            rng,
            baseColor,
            roughness,
            metallic,
            normal,
            wo,
            v4, v3, v2, v1);
    }

    // Save training data
    let index = i * 18;
    data[index + 0] = half(wo.x);
    data[index + 1] = half(wo.y);
    data[index + 2] = half(wo.z);
    data[index + 3] = half(v1.x);
    data[index + 4] = half(v1.y);
    data[index + 5] = half(v1.z);
    data[index + 6] = half(v2.x);
    data[index + 7] = half(v2.y);
    data[index + 8] = half(v2.z);
    data[index + 9] = half(v3.x);
    data[index + 10] = half(v3.y);
    data[index + 11] = half(v3.z);
    data[index + 12] = half(v4.x);
    data[index + 13] = half(v4.y);
    data[index + 14] = half(v4.z);
    data[index + 15] = half(log1p(D.x));
    data[index + 16] = half(log1p(D.y));
    data[index + 17] = half(log1p(D.z));
}
