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
    float mollificationScale;
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
Sampler2D baseColorSampler;

[[vk::binding(3)]]
Sampler2D roughnessMetallicSampler;

[[vk::binding(4)]]
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

func actualBrdfFunc(inout rng: PCG32, wi: float3, wo: float3, uv: float2, mipLevel: float)->float3
{
    const uint ITERATION = 256;

    let mipLevelFloor = floor(mipLevel);

    var brdf = float3(0.0);
    for (var i = 0; i < ITERATION; ++i)
    {
        let uv_modify = addGaussianOffset(rng, uv, mipLevelFloor);

        let baseColor = baseColorSampler.SampleLevel(uv_modify, mipLevelFloor);
        let metallicRoughness =
            roughnessMetallicSampler.SampleLevel(uv_modify, mipLevelFloor);
        var normal = normalSampler.SampleLevel(uv_modify, mipLevelFloor).xyz;
        normal.y = 1.0 - normal.y; // Flip Y coordinate for normal map
        let roughness = metallicRoughness.g;
        let metallic = metallicRoughness.b;

        let N = normalize(normal * 2 - 1);

        let TBN = tangentFrame(N);

        let wi_modify = modifyWi(rng, wi);

        let wi_tangent = mul(TBN, wi_modify);
        let wo_tangent = mul(TBN, wo);
        brdf += DisneyBRDF(wi_tangent, wo_tangent, srgbToLinear(baseColor.rgb), roughness, metallic);
    }

    brdf /= float(ITERATION);
    return brdf;
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

    // sample uv
    let uv = float2(rng.nextFloat(), rng.nextFloat());
    let mipLevel = sampleMipLevel(rng.nextFloat(), computeMipCount(uniforms.textureSize));

    // sample parameters
    let baseColor = srgbToLinear(baseColorSampler.SampleLevel(uv, mipLevel).rgb);
    let metallicRoughness = roughnessMetallicSampler.SampleLevel(uv, mipLevel);
    let normal = normalSampler.SampleLevel(uv, mipLevel).xyz;
    let roughness = metallicRoughness.g;
    let metallic = metallicRoughness.b;

    // compute brdf
    let brdf = actualBrdfFunc(
        rng,
        wi,
        wo,
        uv,
        mipLevel);

    // Save training data
    let index = i * 17;
    data[index + 0] = half(baseColor.r);
    data[index + 1] = half(baseColor.g);
    data[index + 2] = half(baseColor.b);
    data[index + 3] = half(roughness);
    data[index + 4] = half(metallic);
    data[index + 5] = half(normal.x);
    data[index + 6] = half(normal.y);
    data[index + 7] = half(normal.z);
    data[index + 8] = half(wi.x);
    data[index + 9] = half(wi.y);
    data[index + 10] = half(wi.z);
    data[index + 11] = half(wo.x);
    data[index + 12] = half(wo.y);
    data[index + 13] = half(wo.z);
    data[index + 14] = half(log1p(brdf.x));
    data[index + 15] = half(log1p(brdf.y));
    data[index + 16] = half(log1p(brdf.z));
}
