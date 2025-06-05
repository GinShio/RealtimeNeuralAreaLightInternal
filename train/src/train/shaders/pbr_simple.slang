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
