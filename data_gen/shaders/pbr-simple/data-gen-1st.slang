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

        if (wo.z >= 0.0)
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
    mipLevel: float,
    wo: float3,
    v1: float3,
    v2: float3,
    v3: float3,
    v4: float3)
    ->float3
{
    // ────────────────────────────────────────────────────────────────
    const uint ITERATIONS = 1024; // 各手法で 1024 サンプル
    // ────────────────────────────────────────────────────────────────
    let mipLevelFloor = floor(mipLevel);

    float3 D = float3(0.0, 0.0, 0.0);

    // クリッピング & 三角形分割
    let wt = WeightedTriangles.clipAndTriangulateQuad(v1, v2, v3, v4);
    if (wt.triCount == 0 || wt.totalArea <= 0.0)
    {
        return D; // 発光面が見えない
    }

    // ────────────────────────────────────────────────────────────────
    // メインループ：各イテレーションで
    //   ① 面光源サンプル（面積領域）
    //   ② BRDF サンプル（立体角領域）
    // の二手法を評価し、バランスヒューリスティックで MIS 合成する。
    // ────────────────────────────────────────────────────────────────
    for (uint i = 0; i < ITERATIONS; ++i)
    {
        // ────── マテリアル取得（両手法共通で使う） ──────
        let uvM = addGaussianOffset(rng, uv, mipLevelFloor);

        let baseColor = srgbToLinear(baseColorSampler.SampleLevel(uvM, mipLevelFloor).rgb);
        let roughness = roughnessSampler.SampleLevel(uvM, mipLevelFloor).r;
        let metallic = metallicSampler.SampleLevel(uvM, mipLevelFloor).r;

        var normalTex = normalSampler.SampleLevel(uvM, mipLevelFloor).xyz;
        normalTex.y = 1.0 - normalTex.y; // Y 反転
        let N = normalize(normalTex * 2.0 - 1.0);

        // ────────────────────────────────────────────────────────────
        // ① 面光源サンプリング  (面積領域)
        // ────────────────────────────────────────────────────────────
        {
            Triangle tri = wt.sampleTriangle(rng);
            float3 bary = tri.randomBarycentric(rng);
            float3 Lpos = tri.barycentricToPosition(bary); // ライト頂点
            float3 Ln = tri.normal();

            float3 wi = normalize(Lpos);
            float NdotL = saturate(dot(N, wi));
            float LnDotWi = abs(dot(Ln, -wi));
            float dist2 = dot(Lpos, Lpos);

            let G = (NdotL * LnDotWi) / max(dist2, 1e-16);

            if (G > 0.0)
            {
                let brdf = PbrBRDF(baseColor, roughness, metallic, N, wo, wi);

                let pdfLightArea = 1.0 / wt.totalArea;

                let pdfBrdfSolid = PdfPbrBRDF(baseColor, roughness, metallic, N, wo, wi);
                let pdfBrdfArea = pdfBrdfSolid * LnDotWi / max(dist2, 1e-16);

                let wLight = pdfLightArea /
                             max(pdfLightArea + pdfBrdfArea, 1e-16);

                D += brdf * G * wt.totalArea * wLight / float(ITERATIONS);
            }
        }

        // ────────────────────────────────────────────────────────────
        // ② BRDF サンプリング  (立体角領域)
        // ────────────────────────────────────────────────────────────
        {
            float pdfBrdfSolid;
            float3 wi;
            SamplePbrBRDF(
                rng,
                baseColor,
                roughness,
                metallic,
                N,
                wo,
                wi,
                pdfBrdfSolid);

            if (pdfBrdfSolid > 0.0)
            {
                float3 Lpos;
                float3 Ln;
                if (wt.hit(wi, Lpos, Ln))
                {
                    let dist2 = dot(Lpos, Lpos);
                    let LnDotWi = abs(dot(Ln, -wi));
                    let brdf = PbrBRDF(baseColor, roughness, metallic, N, wo, wi);

                    let pdfLightArea = 1.0 / wt.totalArea;
                    let pdfBrdfArea = pdfBrdfSolid * LnDotWi / max(dist2, 1e-16);

                    float wBrdf = pdfBrdfArea /
                                  max(pdfBrdfArea + pdfLightArea, 1e-16);

                    float NdotL = saturate(dot(N, wi));

                    D += brdf * NdotL / max(pdfBrdfSolid, 1e-16) * wBrdf / float(ITERATIONS);
                }
            }
        }
    }

    return D;
}

func log1p4(x: float)->float
{
    var ret = x;
    ret = log(1.0 + ret);
    ret = log(1.0 + ret);
    ret = log(1.0 + ret);
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

    // sample uv
    let uv = float2(rng.nextFloat(), rng.nextFloat());
    let mipLevel = sampleMipLevel(rng.nextFloat(), computeMipCount(uniforms.textureSize));

    // sample parameters
    let baseColor = srgbToLinear(baseColorSampler.SampleLevel(uv, mipLevel).rgb);
    let roughness = roughnessSampler.SampleLevel(uv, mipLevel).r;
    let metallic = metallicSampler.SampleLevel(uv, mipLevel).r;
    var normal = normalSampler.SampleLevel(uv, mipLevel).xyz;
    normal.y = 1.0 - normal.y; // Flip Y coordinate for normal map

    float3 v1;
    float3 v2;
    float3 v3;
    float3 v4;
    sampleDirection(rng, wi, v1, v2, v3, v4);

    // compute D
    let D = computeDistribution(
        rng,
        uv,
        mipLevel,
        wo,
        v1, v2, v3, v4);

    // Save training data
    let index = i * 26;
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
    data[index + 11] = half(v1.x);
    data[index + 12] = half(v1.y);
    data[index + 13] = half(v1.z);
    data[index + 14] = half(v2.x);
    data[index + 15] = half(v2.y);
    data[index + 16] = half(v2.z);
    data[index + 17] = half(v3.x);
    data[index + 18] = half(v3.y);
    data[index + 19] = half(v3.z);
    data[index + 20] = half(v4.x);
    data[index + 21] = half(v4.y);
    data[index + 22] = half(v4.z);
    data[index + 23] = half(log1p4(D.x));
    data[index + 24] = half(log1p4(D.y));
    data[index + 25] = half(log1p4(D.z));
}
