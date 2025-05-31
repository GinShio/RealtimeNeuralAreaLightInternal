const float PI = 3.14159265358979323846;

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
    let H = normalize(V + L);

    let NdotL = saturate(dot(N, L));
    let NdotV = saturate(dot(N, V));
    let NdotH = saturate(dot(N, H));
    let VdotH = saturate(dot(V, H));

    let F0 = lerp(float3(0.04), baseColor, metallic);

    // === specular ===

    // Fresnel (Schlick approximation)
    let F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    // Geometric shadowing (Smith's method)
    let alpha = roughness * roughness;

    let G1V = NdotV / (NdotV * (1.0 - alpha) + alpha);
    let G1L = NdotL / (NdotL * (1.0 - alpha) + alpha);
    let G = G1V * G1L;

    // GGX normal distribution
    let alpha2 = alpha * alpha;
    let denom = (NdotH * NdotH) * (alpha2 - 1.0) + 1.0;
    let D = alpha2 / (PI * denom * denom);

    // Cook-Torrance specular brdf
    let specular = (D * G * F) / (4.0 * NdotL * NdotV + 1e-5);

    // === diffuse ===

    // Normalized Lambert
    let kD = (1.0 - F) * (1.0 - metallic);
    let diffuse = kD * baseColor / PI;

    // === final BRDF ===
    return (diffuse + specular) * NdotL;
}
