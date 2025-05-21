import "base.slang";

struct PushConstants {
  bool fill;
  float3 fillColor;
}

[[push_constant]]
PushConstants pc;

struct VertexOutput
{
  float4 position : SV_POSITION;
  float3 color : COLOR;
  float4 tangent : TANGENT;
  float2 uv : TEXCOORD;
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
  if (pc.fill) {
    return {
      { input.position, 1.0 },
      srgbToLinear(pc.fillColor),
      input.tangent,
      input.uv,
    };
  } else {
    return {
      { input.position, 1.0 },
      srgbToLinear(input.normal),
      input.tangent,
      input.uv,
    };
  }
}

struct FragmentOutput
{
  float4 color : SV_TARGET;
}

[shader("fragment")]
func fsMain(input: VertexOutput) -> FragmentOutput
{
  return {
    { input.color, 1.0 },
  };
}
