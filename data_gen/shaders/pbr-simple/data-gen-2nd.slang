import "../rtxns/Activation";
import "../rtxns/CooperativeVectorAutoDiff";
import "../rtxns/CooperativeVectorDerivatives";
import "../rtxns/CooperativeVectorFunctions";
import "../rtxns/LinearOps";
import "../rtxns/Loss";
import "../rtxns/MLP";
import "../rtxns/PCG32";
import "../rtxns/Utils";

import "brdf";

const float PI = 3.14159265358979323846;

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
    float maxLightSize;
    float maxLightDistance;
}

[[vk::binding(0)]]
uniform ConstantBuffer<UniformBuffer> uniforms;

[[vk::binding(1)]]
RWStructuredBuffer<float> materialData;

[[vk::binding(2)]]
RWStructuredBuffer<float> data;

func localToTangent(v: float3)->float3x3
{
    let up = abs(v.z) < 0.9999 ? float3(0, 0, 1) : float3(1, 0, 0);
    let tangent = normalize(cross(up, v));
    let bitangent = cross(v, tangent);
    return float3x3(tangent, bitangent, v);
}

func sampleWiWo(
    inout rng: PCG32,
    out wi: float3,
    out wo: float3)
{
    for (var i = 0; i < 10; ++i)
    {
        let thetaH = acos(sqrt(1.0 - rng.nextFloat()));
        let phiH = 2.0 * PI * rng.nextFloat();
        let H = float3(
            sin(thetaH) * cos(phiH),
            sin(thetaH) * sin(phiH),
            cos(thetaH));

        let thetaDelta = acos(sqrt(1.0 - rng.nextFloat()));
        let phiDelta = 2.0 * PI * rng.nextFloat();
        let delta = float3(
            sin(thetaDelta) * cos(phiDelta),
            sin(thetaDelta) * sin(phiDelta),
            cos(thetaDelta));

        let M = localToTangent(H);
        let d = mul(M, delta);

        wi = normalize(d);
        wo = 2.0 * dot(wi, H) * H - wi;

        if (wo.z >= 0.0)
        {
            return;
        }
    }
}

func rotationMatrixFromEuler(euler: float3)->float3x3
{
    let cx = cos(euler.x);
    let sx = sin(euler.x);
    let cy = cos(euler.y);
    let sy = sin(euler.y);
    let cz = cos(euler.z);
    let sz = sin(euler.z);

    let rotX = float3x3(
        1.0, 0.0, 0.0,
        0.0, cx, -sx,
        0.0, sx, cx);

    let rotY = float3x3(
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy);

    let rotZ = float3x3(
        cz, -sz, 0.0,
        sz, cz, 0.0,
        0.0, 0.0, 1.0);

    return mul(rotY, mul(rotX, rotZ));
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
    v4: float3,
    area: float)
    ->float3
{
    const uint ITERATIONS = 1024;
    float3 D = float3(0.0, 0.0, 0.0);

    let edge1 = v2 - v1;
    let edge2 = v4 - v1;

    let lightNormal = normalize(cross(edge1, edge2));

    for (uint i = 0; i < ITERATIONS; ++i)
    {
        let u = rng.nextFloat();
        let v = rng.nextFloat();

        let lightPos = v1 + u * edge1 + v * edge2;

        let wi = normalize(lightPos);
        let NdotL = saturate(dot(normal, wi));
        let LnDotWi = saturate(dot(lightNormal, -wi));

        let distanceSquared = dot(lightPos, lightPos);
        let G = (NdotL * LnDotWi) / (distanceSquared + 1e-5);

        if (G > 0.0)
        {
            let brdf = PbrBRDF(baseColor, roughness, metallic, normal, wo, wi);
            D += brdf * G * area;
        }
    }

    return D / float(ITERATIONS);
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
    let baseColor = float3(
        materialData[i * 8 + 0],
        materialData[i * 8 + 1],
        materialData[i * 8 + 2]);
    let roughness = materialData[i * 8 + 3];
    let metallic = materialData[i * 8 + 4];
    let normal = float3(
        materialData[i * 8 + 5],
        materialData[i * 8 + 6],
        materialData[i * 8 + 7]);

    // sample light size
    let width = rng.nextFloat() * uniforms.maxLightSize;
    let height = rng.nextFloat() * uniforms.maxLightSize;
    let size = float2(width, height);
    let area = width * height;

    // sample light rotation
    let x = rng.nextFloat() * 2.0 * PI;
    let y = rng.nextFloat() * 2.0 * PI;
    let z = rng.nextFloat() * 2.0 * PI;
    let rotationEuler = float3(x, y, z);

    // sample light distance
    let distance = rng.nextFloat() * uniforms.maxLightDistance;
    let lightTranslation = normalize(wi) * distance;

    // calculate light vertex positions
    var v1 = float3(-width / 2.0, -height / 2.0, 0.0);
    var v2 = float3(width / 2.0, -height / 2.0, 0.0);
    var v3 = float3(width / 2.0, height / 2.0, 0.0);
    var v4 = float3(-width / 2.0, height / 2.0, 0.0);

    let rotationMatrix = rotationMatrixFromEuler(rotationEuler);

    v1 = mul(rotationMatrix, v1) + lightTranslation;
    v2 = mul(rotationMatrix, v2) + lightTranslation;
    v3 = mul(rotationMatrix, v3) + lightTranslation;
    v4 = mul(rotationMatrix, v4) + lightTranslation;

    // light vertex directions and distances
    let maxDistance =
        uniforms.maxLightDistance + uniforms.maxLightSize / 2.0 * sqrt(2.0);
    let v1Dir = normalize(v1);
    let v1Dist = length(v1) / maxDistance;
    let v2Dir = normalize(v2);
    let v2Dist = length(v2) / maxDistance;
    let v3Dir = normalize(v3);
    let v3Dist = length(v3) / maxDistance;
    let v4Dir = normalize(v4);
    let v4Dist = length(v4) / maxDistance;

    // compute D
    let D = computeDistribution(
        rng,
        baseColor,
        roughness,
        metallic,
        normal,
        wo,
        v1, v2, v3, v4,
        area);

    // Save training data
    let index = i * 23;
    data[index + 0] = wo.x;
    data[index + 1] = wo.y;
    data[index + 2] = wo.z;
    data[index + 3] = v1Dir.x;
    data[index + 4] = v1Dir.y;
    data[index + 5] = v1Dir.z;
    data[index + 6] = v1Dist;
    data[index + 7] = v2Dir.x;
    data[index + 8] = v2Dir.y;
    data[index + 9] = v2Dir.z;
    data[index + 10] = v2Dist;
    data[index + 11] = v3Dir.x;
    data[index + 12] = v3Dir.y;
    data[index + 13] = v3Dir.z;
    data[index + 14] = v3Dist;
    data[index + 15] = v4Dir.x;
    data[index + 16] = v4Dir.y;
    data[index + 17] = v4Dir.z;
    data[index + 18] = v4Dist;
    data[index + 19] = area / (uniforms.maxLightSize * uniforms.maxLightSize);
    data[index + 20] = D.x;
    data[index + 21] = D.y;
    data[index + 22] = D.z;
}
