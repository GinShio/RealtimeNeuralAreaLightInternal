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

func createTbn(normal: float3)
    ->float3x3
{
    var T = float3(1, 0, 0);
    var N = normalize(normal);
    T = normalize(T - dot(T, N) * N);
    let B = normalize(cross(N, T));
    return float3x3(T, B, N);
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
    let N = normal * 2 - 1;

    let TBN = createTbn(N);

    let wi_tangent = mul(TBN, wi);
    let wo_tangent = mul(TBN, wo);
    return DisneyBRDF(wi_tangent, wo_tangent, baseColor.rgb, roughness, metallic);
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
    data[index + 0] = wi.x;
    data[index + 1] = wi.y;
    data[index + 2] = wi.z;
    data[index + 3] = wo.x;
    data[index + 4] = wo.y;
    data[index + 5] = wo.z;
    data[index + 6] = brdf.x;
    data[index + 7] = brdf.y;
    data[index + 8] = brdf.z;
}
