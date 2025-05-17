struct VertexOutput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
}

[shader("vertex")]
func vsMain(uint index: SV_VertexID)->VertexOutput
{
    float2 positions[3] = {
      float2(-1.0, -1.0),
      float2( 3.0, -1.0),
      float2(-1.0, 3.0)
    };

    float2 uvs[3] = {
      float2(0.0, 0.0),
      float2(2.0, 0.0),
      float2(0.0, 2.0)
    };

    VertexOutput output;
    output.position = float4(positions[index], 0.0, 1.0);
    output.uv = uvs[index];
    return output;
}

[[vk::binding(0)]]
Sampler2D inputImage;

struct FragmentOutput
{
    float4 color : SV_TARGET;
}

[shader("fragment")]
func fsMain(VertexOutput input)->FragmentOutput
{
    FragmentOutput output;
    output.color = inputImage.Sample(input.uv);
    return output;
}
