import "../rtxns/PCG32";

public static const float PI = 3.14159265358979323846;

// --- GGX 分布, G1 ----------------------------------------------------------
func D_GGX(float alpha, float NdotH)->float
{
    float a2 = alpha * alpha;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-16);
}

func G1_Smith(float alpha, float NdotV)->float
{
    return NdotV / max(NdotV * (1.0 - alpha) + alpha, 1e-16);
}

// --- PBR BRDF -------------------------------------------------------------
public func PbrBRDF(
    baseColor: float3,
    roughness: float,
    metallic: float,
    normal: float3,
    wo: float3,
    wi: float3, )
    ->float3
{
    let N = normalize(normal);
    let V = normalize(wo);
    let L = normalize(wi);

    if (dot(V, L) < -0.9999)
    {
        // If the light and view direction are exactly opposite, return zero
        // to avoid division by zero in the calculations.
        return float3(0.0, 0.0, 0.0);
    }

    let H = normalize(V + L);

    let NdotL = saturate(dot(N, L));
    let NdotV = saturate(dot(N, V));
    let NdotH = saturate(dot(N, H));
    let VdotH = saturate(dot(V, H));

    if (NdotL <= 0.0 || NdotV <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    // === specular ===

    let F0 = lerp(float3(0.04), baseColor, metallic);

    // Fresnel (Schlick approximation)
    let F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    // Geometric shadowing (Smith's method)
    let alpha = roughness * roughness;

    let G1V = G1_Smith(alpha, NdotV);
    let G1L = G1_Smith(alpha, NdotL);
    let G = G1V * G1L;

    // GGX normal distribution
    let D = D_GGX(alpha, NdotH);

    // Cook-Torrance specular brdf
    let specular = (D * G * F) / max(4.0 * NdotL * NdotV, 1e-16);

    // === diffuse ===

    // Normalized Lambert
    let kD = (1.0 - F) * (1.0 - metallic);
    let diffuse = kD * baseColor / PI;

    // === final BRDF ===
    return diffuse + specular;
}

// --- コサイン重点半球サンプル ---------------------------------------------
func cosineSampleHemisphere(inout rng: PCG32)->float3
{
    float r = sqrt(rng.nextFloat());
    float φ = 2.0 * PI * rng.nextFloat();
    float x = r * cos(φ);
    float y = r * sin(φ);
    float z = sqrt(max(0.0, 1.0 - x * x - y * y));
    return float3(x, y, z); // ローカル空間 (+Z が法線)
}

// --- Heitz 2018 : Visible GGX VNDF サンプル -------------------------------
func sampleGGXVNDF(
    inout rng: PCG32,
    V: float3,
    alpha: float,
    out H: float3,
    out L: float3)
{
    // ストレッチ
    float3 V_stretched = normalize(float3(alpha * V.x, alpha * V.y, V.z));

    // 準備
    float lensq = V_stretched.x * V_stretched.x + V_stretched.y * V_stretched.y;
    float3 T1 = lensq > 0.0 ? float3(-V_stretched.y, V_stretched.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(V_stretched, T1);

    // サンプル
    float r = rng.nextFloat();
    float φ = 2.0 * PI * rng.nextFloat();
    float a = 1.0 / (1.0 + V_stretched.z);
    float P1 = r < a ? r / a : (1.0 - r) / (1.0 - a);
    float P2 = r < a ? (r - a * r) : (r - a) / (1.0 - a);
    float s = sqrt(P1);
    float t = P2 * 0.5 / s;
    float3 Nh = s * cos(φ) * T1 + s * sin(φ) * T2 + sqrt(max(0.0, 1.0 - s * s)) * V_stretched;

    // アンストレッチ ＆ 正規化
    H = normalize(float3(alpha * Nh.x, alpha * Nh.y, max(1e-6, Nh.z)));

    // 反射方向
    L = normalize(2.0 * dot(V, H) * H - V);
}

// rotation from (0, 0, 1) to normal
func tangentFrame(normal: float3)->float3x3
{
    let up = abs(normal.z) < 0.9999 ? float3(0, 0, 1) : float3(1, 0, 0);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return transpose(float3x3(tangent, bitangent, normal));
}

// 返値: サンプリングした wi,   outPdf には p(wi)
public func SamplePbrBRDF(
    inout rng: PCG32,
    baseColor: float3,
    roughness: float,
    metallic: float,
    normal: float3,
    wo: float3,
    out outWi: float3,
    out outPdf: float)
{
    let N = normalize(normal);
    let V = normalize(wo);

    //------------------------------------------------------------------
    // ローブ混合確率
    //------------------------------------------------------------------
    let F0 = lerp(float3(0.04), baseColor, metallic);
    let lumDiff = dot((1.0 - metallic) * baseColor, float3(0.2126, 0.7152, 0.0722));
    let lumSpec = dot(F0, float3(0.2126, 0.7152, 0.0722));
    let wSpec = metallic >= 0.9 ? 1.0 : lumSpec / max(lumDiff + lumSpec, 1e-8);
    let wDiff = 1.0 - wSpec;

    float3 L;
    float pdfSpec = 0.0;
    float pdfDiff = 0.0;

    //------------------------------------------------------------------
    // 1) ローブ選択
    //------------------------------------------------------------------
    if (rng.nextFloat() < wSpec) // 鏡面ローブ
    {
        float alpha = max(1e-4, roughness * roughness);
        float3 H;
        sampleGGXVNDF(rng, V, alpha, H, L);

        float NdotH = saturate(dot(N, H));
        float VdotH = saturate(dot(V, H));
        float G1 = G1_Smith(alpha, saturate(dot(N, V)));
        pdfSpec = D_GGX(alpha, NdotH) * G1 * NdotH / max(4.0 * VdotH, 1e-6);

        pdfDiff = saturate(dot(N, L)) / PI;
    }
    else // 拡散ローブ
    {
        L = cosineSampleHemisphere(rng);

        pdfDiff = saturate(dot(N, L)) / PI;

        float alpha = max(1e-4, roughness * roughness);
        float3 H = normalize(V + L);
        float NdotH = saturate(dot(N, H));
        float VdotH = saturate(dot(V, H));
        float G1 = G1_Smith(alpha, saturate(dot(N, V)));
        pdfSpec = D_GGX(alpha, NdotH) * G1 * NdotH / max(4.0 * VdotH, 1e-6);
    }

    //------------------------------------------------------------------
    // 2) 混合 pdf
    //------------------------------------------------------------------
    outPdf = wSpec * pdfSpec + wDiff * pdfDiff;

    //------------------------------------------------------------------
    // 4) wi の計算
    //------------------------------------------------------------------
    outWi = normalize(mul(tangentFrame(N), L));
}

public func PdfPbrBRDF(
    baseColor: float3,
    roughness: float,
    metallic: float,
    normal: float3,
    wo: float3,
    wi: float3)
    ->float
{
    let N = normalize(normal);
    let V = normalize(wo);
    let L = normalize(wi);

    if (dot(N, L) <= 0.0)
        return 0.0; // 裏面は 0

    // 混合重み
    let F0 = lerp(float3(0.04), baseColor, metallic);
    let lumD = dot((1.0 - metallic) * baseColor, float3(0.2126, 0.7152, 0.0722));
    let lumS = dot(F0, float3(0.2126, 0.7152, 0.0722));
    let wSpec = metallic >= 0.9 ? 1.0 : lumS / max(lumD + lumS, 1e-8);
    let wDiff = 1.0 - wSpec;

    // pdfDiffuse
    let pdfDiff = saturate(dot(N, L)) / PI;

    // pdfSpec
    let alpha = max(1e-4, roughness * roughness);
    let H = normalize(V + L);
    let NdotH = saturate(dot(N, H));
    let VdotH = saturate(dot(V, H));
    let G1 = G1_Smith(alpha, saturate(dot(N, V)));
    let pdfSpec = D_GGX(alpha, NdotH) * G1 * NdotH / max(4.0 * VdotH, 1e-6);

    return wSpec * pdfSpec + wDiff * pdfDiff;
}
