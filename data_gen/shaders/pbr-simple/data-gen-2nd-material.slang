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
Sampler2D roughnessSampler;

[[vk::binding(4)]]
Sampler2D metallicSampler;

[[vk::binding(5)]]
Sampler2D normalSampler;

func computeMipCount(width: int)->int
{
    return int(floor(log2(float(width)))) + 1;
}

func calculateMipLevelAndUv(index: uint, out mipLevel: float, out uv: float2)
{
    var currentMipLevel = 0;
    var currentMipSize = uniforms.textureSize;
    var currentPixelOffset = 0;
    while (currentMipSize > 1)
    {
        if (index < currentPixelOffset + currentMipSize * currentMipSize)
            break;

        currentPixelOffset += currentMipSize * currentMipSize;
        currentMipSize /= 2;
        currentMipLevel++;
    }

    mipLevel = float(currentMipLevel);
    uv.x = float((index - currentPixelOffset) % currentMipSize) / float(currentMipSize);
    uv.y = float((index - currentPixelOffset) / currentMipSize) / float(currentMipSize);
    uv += 0.5 / float(currentMipSize);
}

func srgbToLinear(color: float3)->float3
{
    let useLinear = color <= 0.04045;
    let v1 = color / 12.92;
    let v2 = pow((color + 0.055) / 1.055, 2.4);
    return select(useLinear, v1, v2);
}

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= uniforms.dataSize)
        return;

    // calculate mip level and uv
    float mipLevel;
    float2 uv;
    calculateMipLevelAndUv(i, mipLevel, uv);

    // sample parameters
    let baseColor = srgbToLinear(baseColorSampler.SampleLevel(uv, mipLevel).rgb);
    // let roughness = roughnessSampler.SampleLevel(uv, mipLevel).r;
    let roughness = 0.001;
    // let metallic = metallicSampler.SampleLevel(uv, mipLevel).r;
    let metallic = 1.0;
    var normal = normalSampler.SampleLevel(uv, mipLevel).xyz;
    normal.y = 1.0 - normal.y; // Flip Y coordinate for normal map

    // Save training data
    let index = i * 8;
    data[index + 0] = half(baseColor.r);
    data[index + 1] = half(baseColor.g);
    data[index + 2] = half(baseColor.b);
    data[index + 3] = half(roughness);
    data[index + 4] = half(metallic);
    data[index + 5] = half(normal.x);
    data[index + 6] = half(normal.y);
    data[index + 7] = half(normal.z);
}
