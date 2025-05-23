func lambda_G(float cosTheta, float alpha) -> float
{
    let a = alpha * sqrt(max(0.0, 1.0 - cosTheta * cosTheta)) / max(cosTheta, 1e-5);
    return (-1.0 + sqrt(1.0 + a * a)) / 2.0;
}

func DisneyBRDF(
  float3 baseColor,
  // float subsurface,
  float metallic,
  // float specular,
  // float specularTint,
  float roughness,
  // float anisotropic,
  // float sheen,
  // float sheenTint,
  // float clearcoat,
  // float clearcoatGloss,
  float4 Tangent,
  float3 View,
  float3 Light
) -> float3
{
  let subsurface = 0.0;
  let specular = 0.5;
  let specularTint = 0.0;
  let anisotropic = 0.0;
  let sheen = 0.0;
  let sheenTint = 0.0;
  let clearcoat = 0.0;
  let clearcoatGloss = 0.0;


  const float PI = 3.14159265358979323846;

  let N = float3(0, 1, 0);
  let V = normalize(View);
  let L = normalize(Light);
  let T = normalize(Tangent.xyz);
  let B = normalize(cross(N, T) * Tangent.w);  // Bitangent

  let H = normalize(L + V); // Half vector

  let NdotL = saturate(dot(N, L));
  let NdotV = saturate(dot(N, V));
  let NdotH = saturate(dot(N, H));
  let LdotH = saturate(dot(L, H));
  let HdotX = dot(H, T);
  let HdotY = dot(H, B);

  //----------------------------------------
  // Specular F0 with tint
  //----------------------------------------
  let dielectricSpecular = 0.08 * specular;
  var F0 = lerp(float3(dielectricSpecular), baseColor, metallic);

  let luminance = max(1e-5, dot(baseColor, float3(0.3, 0.6, 0.1)));
  let tint = baseColor / luminance;
  F0 = lerp(F0, F0 * tint, specularTint * (1.0 - metallic));

  //----------------------------------------
  // Anisotropic Specular Distribution (GTR2)
  //----------------------------------------
  let aspect = sqrt(1.0 - 0.9 * anisotropic);
  let ax = max(0.001, (roughness * roughness) / aspect);
  let ay = max(0.001, (roughness * roughness) * aspect);

  let HdotN = dot(H, N);
  let D_denom = HdotX * HdotX / (ax * ax) + HdotY * HdotY / (ay * ay) + HdotN * HdotN;
  let D = 1.0 / (PI * ax * ay * D_denom * D_denom);

  //----------------------------------------
  // Geometry (Smith)
  //----------------------------------------
  let G = 1.0 / (1.0 + lambda_G(NdotL, roughness) + lambda_G(NdotV, roughness));

  //----------------------------------------
  // Fresnel (Schlick)
  //----------------------------------------
  let F = F0 + (1.0 - F0) * pow(1.0 - LdotH, 5.0);

  //----------------------------------------
  // Specular Term
  //----------------------------------------
  let specularBRDF = D * G * F / max(4.0 * NdotL * NdotV, 1e-5);

  //----------------------------------------
  // Diffuse + Subsurface
  //----------------------------------------
  let FL = pow(1.0 - NdotL, 5.0);
  let FV = pow(1.0 - NdotV, 5.0);
  let FD90 = 0.5 + 2.0 * LdotH * LdotH * roughness;
  var Fd = baseColor * (1.0 / PI) * ((1.0 + (FD90 - 1.0) * FL) * (1.0 + (FD90 - 1.0) * FV)) * (1.0 - metallic);

  let ss90 = roughness * LdotH * LdotH;
  let Fss = baseColor * (1.0 / PI) * (1.0 + ss90 * (1.0 - FL)) * (1.0 + ss90 * (1.0 - FV));
  Fd = lerp(Fd, Fss, subsurface);

  //----------------------------------------
  // Sheen
  //----------------------------------------
  let sheenColor = lerp(float3(1.0, 1.0, 1.0), tint, sheenTint);
  let sheenBRDF = sheen * sheenColor * pow(1.0 - LdotH, 5.0) * (1.0 - metallic);

  //----------------------------------------
  // Clearcoat
  let alpha_c = lerp(0.1, 0.001, clearcoatGloss);
  let dc_denom = NdotH * NdotH * (alpha_c * alpha_c - 1.0) + 1.0;
  let Dc = (alpha_c * alpha_c - 1.0) / (PI * log(alpha_c * alpha_c) * dc_denom * dc_denom);
  let Fc = 0.04 + (1.0 - 0.04) * pow(1.0 - LdotH, 5.0);
  let Gc = G; // reuse
  let clearBRDF = clearcoat * Dc * Fc * Gc / max(4.0 * NdotL * NdotV, 1e-5);

  //----------------------------------------
  // Final BRDF sum
  //----------------------------------------
  return Fd + specularBRDF + sheenBRDF + float3(clearBRDF);
}
