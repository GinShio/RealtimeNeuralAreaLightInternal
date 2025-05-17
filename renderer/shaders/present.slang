struct VertexInput
{
    uint index: SV_VertexID;
}

struct VertexOutput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
}

[shader("vertex")]
func vsMain(input: VertexInput)->VertexOutput
{
    float2 positions[] = {
      {-1.0, -1.0},
      { 3.0, -1.0},
      {-1.0, 3.0},
    };

    float2 uvs[] = {
      {0.0, 0.0},
      {2.0, 0.0},
      {0.0, 2.0},
    };

    return {
        float4(positions[input.index], 0.0, 1.0),
        uvs[input.index],
    };
}

[[vk::binding(0)]]
Sampler2D inputImage;

struct FragmentOutput
{
    float4 color : SV_TARGET;
}

[shader("fragment")]
func fsMain(input: VertexOutput )->FragmentOutput
{
  return {
    inputImage.Sample(input.uv)
  };
}
