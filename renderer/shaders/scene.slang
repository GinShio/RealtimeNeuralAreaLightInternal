struct VertexInput
{
    float2 position;
    float3 color;
}

struct VertexOutput
{
    float4 position : SV_POSITION;
    float3 color : TEXCOORD0;
}

[shader("vertex")]
func vsMain(input: VertexInput)->VertexOutput
{
    return {
        { input.position, 0.0, 1.0 },
        input.color,
    };
}

struct FragmentOutput
{
    float4 color : SV_TARGET;
}

[shader("fragment")]
func fsMain(input: VertexOutput)->FragmentOutput
{
    return {
        { input.color, 1.0 },
    };
}
