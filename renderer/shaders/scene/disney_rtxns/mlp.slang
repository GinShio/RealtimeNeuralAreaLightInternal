import "../base.slang";

import "../../rtxns/CooperativeVectorFunctions";
import "../../rtxns/Utils";
import "../../rtxns/LinearOps";

[[vk::binding(0, 2)]]
StructuredBuffer<half> params;

struct LightParams {
  float3 position;
  float intensity;
  float3 color;
  uint padding;
}

struct Uniforms {
  uint4 weight_offsets;
  uint4 bias_offsets;

  float4x4 view;
  float4x4 projection;

  uint samplerIndex;
  uint light_count;

  LightParams light[3];
}

[[vk::binding(0, 3)]]
uniform ConstantBuffer<Uniforms> uniforms;


struct PushConstants {
  float4x4 model;
  uint baseColorIndex;
  uint normalIndex;
  uint metallicRoughnessIndex;
  uint padding;
  float3 baseColorOverrideValue;
  uint baseColorOverride;
  float roughnessOverrideValue;
  uint roughnessOverride;
  float metallicOverrideValue;
  uint metallicOverride;
}

[[push_constant]]
PushConstants pc;

struct VertexOutput
{
  float4 position : SV_POSITION;
  float3 viewPosition : TEXCOORD0;
  float3 viewNormal : TEXCOORD1;
  float4 viewTangent : TEXCOORD2;
  float2 uv : TEXCOORD3;
}

float3x3 inverse3x3(float3x3 m)
{
  float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
  float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
  float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

  float det = a00 * (a11 * a22 - a12 * a21) -
    a01 * (a10 * a22 - a12 * a20) +
    a02 * (a10 * a21 - a11 * a20);

  if (abs(det) < 1e-6) return float3x3(0.0); // 非正則行列

  float invDet = 1.0 / det;

  return float3x3(
    +(a11 * a22 - a12 * a21) * invDet,
    -(a01 * a22 - a02 * a21) * invDet,
    +(a01 * a12 - a02 * a11) * invDet,

    -(a10 * a22 - a12 * a20) * invDet,
    +(a00 * a22 - a02 * a20) * invDet,
    -(a00 * a12 - a02 * a10) * invDet,

    +(a10 * a21 - a11 * a20) * invDet,
    -(a00 * a21 - a01 * a20) * invDet,
    +(a00 * a11 - a01 * a10) * invDet
  );
}

func srgbToLinear(color: float3) -> float3
{
  let useLinear = color <= 0.04045;
  let v1 = color / 12.92;
  let v2 = pow((color + 0.055) / 1.055, 2.4);
  return select(useLinear, v1, v2);
}

[shader("vertex")]
func vsMain(input: VertexInput) -> VertexOutput
{
  let viewModel = mul(uniforms.view, pc.model);
  let viewModelIT = inverse3x3(transpose(float3x3(viewModel)));

  let viewPosition = mul(viewModel, float4(input.position, 1.0));

  let viewNormal = normalize(mul(viewModelIT, input.normal));

  let viewTangent = normalize(mul(viewModelIT, input.tangent.xyz));
  let viewTangentW = input.tangent.w;

  let uv = input.uv;

  let position = mul(uniforms.projection, viewPosition);

  return {
      position,
      viewPosition.xyz,
      viewNormal.xyz,
      float4(viewTangent.xyz, viewTangentW),
      uv,
  };
}

func normalMap(normal: float3, tangent: float4, normalMapValue: float3) -> float3
{
  let N = normalize(normal);
  let T = normalize(tangent.xyz - dot(tangent.xyz, N) * N);
  let B = cross(N, T) * tangent.w;

  let TBN = transpose(float3x3(T, B, N));
  let n = normalMapValue * 2.0 - 1.0;
  let result = mul(TBN, n);

  return normalize(result);
}

struct FragmentOutput
{
  float4 color : SV_TARGET;
}

[shader("fragment")]
func fsMain(input: VertexOutput) -> FragmentOutput
{
  let sampler = samplerList[uniforms.samplerIndex];

  var baseColor = float3(1.0, 1.0, 1.0);
  if (pc.baseColorIndex != 0xFFFFFFFF)
  {
    let baseColorTexture = textures[pc.baseColorIndex];
    baseColor = baseColorTexture.Sample(sampler, input.uv).rgb;
  }
  var normal = float3(0.5, 0.5, 1.0);
  if (pc.normalIndex != 0xFFFFFFFF)
  {
    let normalTexture = textures[pc.normalIndex];
    normal = normalTexture.Sample(sampler, input.uv).rgb;
  }
  var metallic = 1.0;
  var roughness = 0.1;
  if (pc.metallicRoughnessIndex != 0xFFFFFFFF)
  {
    let metallicRoughnessTexture = textures[pc.metallicRoughnessIndex];
    let metallicRoughness = metallicRoughnessTexture.Sample(sampler, input.uv).rgb;
    roughness = metallicRoughness.g;
    metallic = metallicRoughness.b;
  }

  if (pc.baseColorOverride == 1)
  {
    baseColor = pc.baseColorOverrideValue;
  }
  if (pc.roughnessOverride == 1)
  {
    roughness = pc.roughnessOverrideValue;
  }
  if (pc.metallicOverride == 1)
  {
    metallic = pc.metallicOverrideValue;
  }

  let specular = 0.5;

  let V = normalize(-input.viewPosition);
  let N = normalMap(input.viewNormal, input.viewTangent, normal);

  var color = float3(0.0, 0.0, 0.0);
  for (var i = 0; i < uniforms.light_count; i++)
  {
    let light = uniforms.light[i];

    let lightViewPosition = mul(uniforms.view, float4(light.position, 1.0)).xyz;
    let L = normalize(lightViewPosition - input.viewPosition);
    let H = normalize(L + V);

    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let LdotH = max(dot(L, H), 0.0);

    CoopVec<half, 5> inputParams;
    CoopVec<half, 64> hiddenParams;
    CoopVec<half, 4> outputParams;

    inputParams = { NdotL, NdotV, NdotH, LdotH, roughness };

    hiddenParams = rtxns::LinearOp<half, 64, 5>(
      inputParams,
      params,
      uniforms.weight_offsets[0],
      uniforms.bias_offsets[0],
      CoopVecMatrixLayout::InferencingOptimal,
      CoopVecComponentType::Float16
    );
    hiddenParams = rtxns::relu(hiddenParams);

    hiddenParams = rtxns::LinearOp<half, 64, 64>(
      hiddenParams,
      params,
      uniforms.weight_offsets[1],
      uniforms.bias_offsets[1],
      CoopVecMatrixLayout::InferencingOptimal,
      CoopVecComponentType::Float16
    );
    hiddenParams = rtxns::relu(hiddenParams);

    hiddenParams = rtxns::LinearOp<half, 64, 64>(
      hiddenParams,
      params,
      uniforms.weight_offsets[2],
      uniforms.bias_offsets[2],
      CoopVecMatrixLayout::InferencingOptimal,
      CoopVecComponentType::Float16
    );
    hiddenParams = rtxns::relu(hiddenParams);

    outputParams = rtxns::LinearOp<half, 4, 64>(
      hiddenParams,
      params,
      uniforms.weight_offsets[3],
      uniforms.bias_offsets[3],
      CoopVecMatrixLayout::InferencingOptimal,
      CoopVecComponentType::Float16
    );
    outputParams = exp(outputParams);

    let outParams = float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);

    // Calculate final color
    float3 Cdlin = float3(pow(baseColor[0], 2.2), pow(baseColor[1], 2.2), pow(baseColor[2], 2.2));
    float3 Cspec0 = lerp(specular * .08 * float3(1), Cdlin, metallic);
    float3 brdfn = outParams.x * Cdlin * (1 - metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = max(brdfn * float3(NdotL), 0.0) * light.intensity * light.color;

    color += (NdotL > 0.0 && NdotV > 0.0)? colorh : 0.0;
  }

  return {
    { color, 1.0 },
  };
}
