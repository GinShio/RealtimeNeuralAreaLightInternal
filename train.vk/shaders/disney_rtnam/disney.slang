const static float PI = 3.14159265358979323846;

float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

float Gtr1(float NdotH, float a)
{
    if (a >= 1)
    {
        return 1 / PI;
    }
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

float Gtr2(float NdotH, float ax)
{
    float a = ax * (1 / ax / ax * (1 - NdotH * NdotH) + NdotH * NdotH);
    return 1 / (PI * a * a);
}

float SmithGGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

float SmithGGXAnisotropy(float NdotV, float ax)
{
    return 1 / (NdotV + sqrt(ax * ax * (1 - NdotV * NdotV) + NdotV * NdotV));
}

float4 Disney(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - .5f) + .5f);

    // specular
    float ax = max(.001f, roughness * roughness);
    float Ds = Gtr2(NdotH, ax);
    float FH = SchlickFresnel(LdotH);
    float Gs = SmithGGXAnisotropy(NdotL, ax);
    Gs *= SmithGGXAnisotropy(NdotV, ax);

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = Gtr1(NdotH, .01f);
    float Fr = lerp(.04f, 1.0f, FH);
    float Gr = SmithGGX(NdotL, .25f) * SmithGGX(NdotV, .25f);

    return float4((1 / PI) * ss, Gs * Ds, FH, .25 * Gr * Fr * Dr);
}

func DisneyBRDF(wi: float3, wo: float3, baseColor: float3, roughness: float, metallic: float)->float3
{
    let specular = 0.5;

    let NdotL = max(wi.z, 0.0);
    let NdotV = max(wo.z, 0.0);

    if (NdotL <= 0 || NdotV <= 0)
        return float3(0, 0, 0);

    let H = normalize(wi + wo);
    let NdotH = max(H.z, 0.0);
    let LdotH = max(dot(wi, H), 0.0);

    let disneyParams = Disney(NdotL, NdotV, NdotH, LdotH, roughness);

    let C_specular = lerp(specular * 0.08 * 1.0, baseColor, metallic);
    let brdf = disneyParams.x * baseColor * (1.0 - metallic) + disneyParams.y * lerp(C_specular, 1.0, disneyParams.z) + disneyParams.w;

    return brdf;
}
