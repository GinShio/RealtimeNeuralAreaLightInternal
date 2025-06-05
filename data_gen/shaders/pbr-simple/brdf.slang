import "../rtxns/PCG32";

public static const float PI = 3.14159265358979323846;

// ============================================================
// GGX 分布 D と Smith G1（Schlick 近似用の k を受け取る）
// ============================================================

//  alpha : roughness² （RMS スロープ）を想定
float D_GGX(float alpha, float NdotH)
{
    float a2 = alpha * alpha; // = roughness^4
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-16);
}

//  k : Smith-Schlick の k（例: ((α+1)²)/8）
float G1_Schlick(float k, float NdotV)
{
    return NdotV / max(NdotV * (1.0 - k) + k, 1e-16);
}

// ============================================================
// Cook-Torrance PBR BRDF（ラジアンスを返す）
// ============================================================
public float3 PbrBRDF(
    float3 baseColor,
    float roughness,
    float metallic,
    float3 normal,
    float3 wo, // 視線 (outgoing) : 外向き
    float3 wi) // 光線 (incoming) : 外向き
{
    float3 N = normalize(normal);
    float3 V = normalize(wo);
    float3 L = normalize(wi);

    // 表面裏側は寄与 0
    float NdotV = dot(N, V);
    float NdotL = dot(N, L);
    if (NdotV <= 0.0 || NdotL <= 0.0)
        return float3(0.0, 0.0, 0.0);

    if (dot(V, L) < -0.99999)
    {
        // If the light and view direction are exactly opposite, return zero
        // to avoid division by zero in the calculations.
        return float3(0.0, 0.0, 0.0);
    }

    // 半ベクトル
    float3 H = normalize(V + L);
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    // ----------- 鏡面項 --------------------------------------
    float alpha = max(1e-4, roughness * roughness);
    float k = (alpha + 1.0) * (alpha + 1.0) * 0.125; // Smith-Schlick

    // フレネル（Schlick）
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    float D = D_GGX(alpha, NdotH);
    float G = G1_Schlick(k, NdotV) * G1_Schlick(k, NdotL);

    float3 specular = (D * G * F) / max(4.0 * NdotL * NdotV, 1e-16);

    // ----------- 拡散項 --------------------------------------
    float3 kD = (1.0 - F) * (1.0 - metallic);
    float3 diffuse = kD * baseColor / PI;

    // ----------- 合成 ----------------------------------------
    return diffuse + specular;
}

//--------------------------------------------------------------
// 接ベクトルフレーム（ローカル +Z → normal）
//--------------------------------------------------------------
float3x3 tangentFrame(float3 normal)
{
    float3 up = abs(normal.z) < 0.9999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return transpose(float3x3(tangent, bitangent, normal)); // 列→行
}

//--------------------------------------------------------------
// コサイン重点半球サンプリング（ローカル空間）
//--------------------------------------------------------------
float3 cosineSampleHemisphere(inout PCG32 rng)
{
    float r = sqrt(rng.nextFloat());
    float φ = 2.0 * PI * rng.nextFloat();
    float x = r * cos(φ);
    float y = r * sin(φ);
    float z = sqrt(max(0.0, 1.0 - r * r));
    return float3(x, y, z); // +Z が法線
}

//--------------------------------------------------------------
// Heitz 2018 : Visible GGX VNDF サンプル（ローカル空間）
//--------------------------------------------------------------
void sampleGGXVNDF(
    inout PCG32 rng,
    float3 V, // 視線 (local)
    float alpha,
    out float3 H, // 半ベクトル (local)
    out float3 L) // 反射方向 (local)
{
    // 1. 視線をストレッチ
    float3 Vh = normalize(float3(alpha * V.x, alpha * V.y, V.z));

    // 2. 直交基底
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq)
                            : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);

    // 3. 単位円盤上をサンプリングし可視領域へ射影
    float U1 = rng.nextFloat();
    float U2 = rng.nextFloat();
    float r = sqrt(U1);
    float phi = 2.0 * PI * U2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);

    float s = 0.5 * (1.0 + Vh.z);
    t2 = lerp(sqrt(max(0.0, 1.0 - t1 * t1)), t2, s);

    // 4. ストレッチ空間の半ベクトル
    float3 Nh = t1 * T1 + t2 * T2 +
                sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // 5. アンストレッチ & 正規化
    H = normalize(float3(alpha * Nh.x, alpha * Nh.y, max(1e-6, Nh.z)));

    // 6. 反射方向
    L = normalize(2.0 * dot(V, H) * H - V);
}

//--------------------------------------------------------------
// PBR BRDF サンプリング : 鏡面 + 拡散の MIS
//--------------------------------------------------------------
public void SamplePbrBRDF(
    inout PCG32 rng,
    float3 baseColor,
    float roughness,
    float metallic,
    float3 normal,    // (world)
    float3 wo,        // 視線(outgoing, world)
    out float3 outWi, // 入射方向 Wi (world)
    out float outPdf)
{
    // ––––– 変換行列
    float3x3 tangentFrameMatrix = tangentFrame(normal);
    float3x3 tangentFrameMatrixInv = transpose(tangentFrameMatrix);

    // ––––– ローカル空間へ
    float3 V = normalize(mul(tangentFrameMatrix, wo)); // 視線 (local)
    float3 Nt = float3(0, 0, 1);                       // 法線 (local)

    // ––––– ローブ混合確率
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float lumDiff = dot((1.0 - metallic) * baseColor, float3(0.2126, 0.7152, 0.0722));
    float lumSpec = dot(F0, float3(0.2126, 0.7152, 0.0722));
    float wSpec = metallic >= 0.9 ? 1.0 : lumSpec / max(lumDiff + lumSpec, 1e-8);
    float wDiff = 1.0 - wSpec;

    // ––––– 共有パラメータ
    float alpha = max(1e-4, roughness * roughness);
    float k = (alpha + 1.0) * (alpha + 1.0) * 0.125; // Smith-Schlick

    float3 L;
    float pdfSpec = 0.0;
    float pdfDiff = 0.0;

    // ––––– 1) ローブ選択
    if (rng.nextFloat() < wSpec) // 鏡面ローブ
    {
        float3 H;
        sampleGGXVNDF(rng, V, alpha, H, L);

        float NdotH = saturate(dot(Nt, H));
        float NdotV = saturate(dot(Nt, V));
        float G1 = G1_Schlick(k, NdotV);

        pdfSpec = D_GGX(alpha, NdotH) * G1 * NdotH /
                  max(4.0 * NdotV, 1e-6);

        pdfDiff = saturate(L.z) / PI; // Nt·L == L.z
    }
    else // 拡散ローブ
    {
        L = cosineSampleHemisphere(rng); // ← 関数で生成
        pdfDiff = L.z / PI;

        float3 H = normalize(V + L);
        float NdotH = saturate(dot(Nt, H));
        float NdotV = saturate(dot(Nt, V));
        float G1 = G1_Schlick(k, NdotV);

        pdfSpec = D_GGX(alpha, NdotH) * G1 * NdotH /
                  max(4.0 * NdotV, 1e-6);
    }

    // ––––– バックフェース棄却
    if (L.z <= 0.0)
    {
        outPdf = 0.0;
        outWi = float3(0, 0, 0);
        return;
    }

    // ––––– 2) 混合 PDF とワールド変換
    outPdf = wSpec * pdfSpec + wDiff * pdfDiff;
    outWi = mul(tangentFrameMatrixInv, L);

    // outPdf = 1.0;
    // outWi = -reflect(wo, normal); // 入射方向 Wi (local)
}

// ============================================================
// BRDF サンプラーに一致する PDF（MIS 用）
// ============================================================
public float PdfPbrBRDF(
    float3 baseColor,
    float roughness,
    float metallic,
    float3 normal,
    float3 wo,
    float3 wi)
{
    float3 N = normalize(normal);
    float3 V = normalize(wo);
    float3 L = normalize(wi);

    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL <= 0.0 || NdotV <= 0.0)
        return 0.0; // 裏面は 0

    // ----------- ローブ混合重み（輝度ベース） -----------------
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float lumD = dot((1.0 - metallic) * baseColor, float3(0.2126, 0.7152, 0.0722));
    float lumS = dot(F0, float3(0.2126, 0.7152, 0.0722));
    float wSpec = metallic >= 0.9 ? 1.0 : lumS / max(lumD + lumS, 1e-8);
    float wDiff = 1.0 - wSpec;

    // ----------- 拡散 PDF ------------------------------------
    float pdfDiff = NdotL / PI;

    // ----------- 鏡面 PDF ------------------------------------
    float alpha = max(1e-4, roughness * roughness);
    float k = (alpha + 1.0) * (alpha + 1.0) * 0.125;

    float3 H = normalize(V + L);
    float NdotH = saturate(dot(N, H));

    float G1 = G1_Schlick(k, NdotV);
    float pdfSpec = D_GGX(alpha, NdotH) * G1 * NdotH /
                    max(4.0 * NdotV, 1e-6); // ← N·V を使用

    // ----------- 混合 PDF ------------------------------------
    return wSpec * pdfSpec + wDiff * pdfDiff;
}
