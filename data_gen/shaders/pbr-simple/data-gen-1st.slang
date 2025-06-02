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
    float mollificationScale;
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
Sampler2D baseColorSampler;

[[vk::binding(3)]]
Sampler2D roughnessSampler;

[[vk::binding(4)]]
Sampler2D metallicSampler;

[[vk::binding(5)]]
Sampler2D normalSampler;

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

func computeMipCount(width: int)->int
{
    return int(floor(log2(float(width)))) + 1;
}

func sampleMipLevel(u: float, mipCount: int)->float
{
    let bias = log(2.0);
    let l_max = float(mipCount - 1);
    let l = -log(1.0 - u) / bias;
    return clamp(l, 0.0, l_max);
}

func srgbToLinear(color: float3)->float3
{
    let useLinear = color <= 0.04045;
    let v1 = color / 12.92;
    let v2 = pow((color + 0.055) / 1.055, 2.4);
    return select(useLinear, v1, v2);
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

func modifyWi(inout rng: PCG32, wi: float3)->float3
{
    if (pc.mollificationScale <= 0.0)
    {
        return wi;
    }

    let deg = 10.0 * pc.mollificationScale;
    let cosThetaMax = cos(radians(deg));

    let cosTheta = 1.0 - rng.nextFloat() * (1.0 - cosThetaMax);
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    let phi = 2.0 * PI * rng.nextFloat();
    let localDir = float3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta);

    let TBN = transpose(tangentFrame(wi));
    let modifiedWi = mul(TBN, localDir);

    return normalize(modifiedWi);
}

func addGaussianOffset(inout rng: PCG32, uv: float2, mipLevel: float)->float2
{
    let levelWidth = uniforms.textureSize / exp2(mipLevel);
    let pixelSize = 1.0 / levelWidth;
    let sigma = 0.5 * 0.5 * pixelSize;

    // Box-Muller method for generating Gaussian noise
    let r = sqrt(-2.0 * log(max(rng.nextFloat(), 1e-6)));
    let theta = 2.0 * PI * rng.nextFloat();
    let offset = sigma * r * float2(cos(theta), sin(theta));

    return uv + offset;
}

func computeDistribution(
    inout rng: PCG32,
    uv: float2,
    wo: float3,
    v1: float3,
    v2: float3,
    v3: float3,
    v4: float3,
    area: float,
    mipLevel: float)
    ->float3
{
    const uint ITERATIONS = 1024;

    let mipLevelFloor = floor(mipLevel);

    float3 D = float3(0.0, 0.0, 0.0);

    let edge1 = v2 - v1;
    let edge2 = v4 - v1;

    let lightNormal = normalize(cross(edge1, edge2));

    for (uint i = 0; i < ITERATIONS; ++i)
    {
        let uv_modify = addGaussianOffset(rng, uv, mipLevelFloor);

        let baseColor = baseColorSampler.SampleLevel(uv_modify, mipLevelFloor);
        let roughness = roughnessSampler.SampleLevel(uv_modify, mipLevelFloor).r;
        let metallic = metallicSampler.SampleLevel(uv_modify, mipLevelFloor).r;
        var normal = normalSampler.SampleLevel(uv_modify, mipLevelFloor).xyz;
        normal.y = 1.0 - normal.y; // Flip Y coordinate for normal map

        let u = rng.nextFloat();
        let v = rng.nextFloat();

        let lightPos = v1 + u * edge1 + v * edge2;

        let wi = normalize(lightPos);
        let NdotL = saturate(dot(normal, wi));
        let LnDotWi = saturate(dot(lightNormal, -wi));

        let distanceSquared = dot(lightPos, lightPos);
        let G = (NdotL * LnDotWi) / (distanceSquared + 1e-5);

        let wi_modify = modifyWi(rng, wi);

        if (G > 0.0)
        {
            let brdf = PbrBRDF(srgbToLinear(baseColor.rgb), roughness, metallic, normal, wo, wi_modify);
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

    // sample uv
    let uv = float2(rng.nextFloat(), rng.nextFloat());
    let mipLevel = sampleMipLevel(rng.nextFloat(), computeMipCount(uniforms.textureSize));

    // sample parameters
    let baseColor = srgbToLinear(baseColorSampler.SampleLevel(uv, mipLevel).rgb);
    let roughness = roughnessSampler.SampleLevel(uv, mipLevel).r;
    let metallic = metallicSampler.SampleLevel(uv, mipLevel).r;
    var normal = normalSampler.SampleLevel(uv, mipLevel).xyz;
    normal.y = 1.0 - normal.y; // Flip Y coordinate for normal map

    // sample light size
    let width = rng.nextFloat() * uniforms.maxLightSize;
    let height = rng.nextFloat() * uniforms.maxLightSize;
    let size = float2(width, height);
    let area = width * height;
    let normalizedArea = area / (uniforms.maxLightSize * uniforms.maxLightSize);

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
        uv,
        wo,
        v1, v2, v3, v4,
        area,
        mipLevel);

    // Save training data
    let index = i * 31;
    data[index + 0] = half(baseColor.x);
    data[index + 1] = half(baseColor.y);
    data[index + 2] = half(baseColor.z);
    data[index + 3] = half(roughness);
    data[index + 4] = half(metallic);
    data[index + 5] = half(normal.x);
    data[index + 6] = half(normal.y);
    data[index + 7] = half(normal.z);
    data[index + 8] = half(wo.x);
    data[index + 9] = half(wo.y);
    data[index + 10] = half(wo.z);
    data[index + 11] = half(v1Dir.x);
    data[index + 12] = half(v1Dir.y);
    data[index + 13] = half(v1Dir.z);
    data[index + 14] = half(v1Dist);
    data[index + 15] = half(v2Dir.x);
    data[index + 16] = half(v2Dir.y);
    data[index + 17] = half(v2Dir.z);
    data[index + 18] = half(v2Dist);
    data[index + 19] = half(v3Dir.x);
    data[index + 20] = half(v3Dir.y);
    data[index + 21] = half(v3Dir.z);
    data[index + 22] = half(v3Dist);
    data[index + 23] = half(v4Dir.x);
    data[index + 24] = half(v4Dir.y);
    data[index + 25] = half(v4Dir.z);
    data[index + 26] = half(v4Dist);
    data[index + 27] = half(normalizedArea);
    data[index + 28] = half(D.x);
    data[index + 29] = half(D.y);
    data[index + 30] = half(D.z);
}
