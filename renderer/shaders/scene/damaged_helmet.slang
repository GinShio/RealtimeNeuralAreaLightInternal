struct PushConstants {
  float4x4 model;
  float4x4 view;
  float4x4 projection;
}

[[push_constant]]
PushConstants pc;

struct VertexInput
{
  float3 position;
  float3 normal;
  float4 tangent;
  float2 uv;
}

struct VertexOutput
{
  float4 position : SV_POSITION;
  float3 viewPosition : TEXCOORD0;
  float3 viewNormal : TEXCOORD1;
  float4 viewTangent : TEXCOORD2;
  float2 uv : TEXCOORD3;
}

func srgbToLinear(color: float3) -> float3
{
  let useLinear = color <= 0.04045;
  let v1 = color / 12.92;
  let v2 = pow((color + 0.055) / 1.055, 2.4);
  return select(useLinear, v1, v2);
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


[shader("vertex")]
func vsMain(input: VertexInput) -> VertexOutput
{
  let viewModel = mul(pc.view, pc.model);
  let viewModelIT = inverse3x3(transpose(float3x3(viewModel)));

  let viewPosition = mul(viewModel, float4(input.position, 1.0));

  let viewNormal = normalize(mul(viewModelIT, input.normal));

  let viewTangent = normalize(mul(viewModelIT, input.tangent.xyz));
  let viewTangentW = input.tangent.w;

  let uv = input.uv;

  let position = mul(pc.projection, viewPosition);

  return {
      position,
      viewPosition.xyz,
      viewNormal.xyz,
      float4(viewTangent.xyz, viewTangentW),
      uv,
  };
}

struct FragmentOutput
{
  float4 color : SV_TARGET;
}

[shader("fragment")]
func fsMain(input: VertexOutput) -> FragmentOutput
{
  let normal = normalize(input.viewNormal);
  let normalColor = srgbToLinear(0.5 * (normal + 1.0));
  return {
      { normalColor, 1.0 },
  };
}
