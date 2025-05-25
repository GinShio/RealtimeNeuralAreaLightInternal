[[vk::binding(0)]]
Texture2D<float4> inputTexture;
[[vk::binding(1)]]
RWTexture2D<float4> outputTexture;

func linearToSRGB(color: float3)->float3
{
    let useLinear = color <= 0.0031308;
    let v1 = 12.92 * color;
    let v2 = 1.055 * pow(color, 1.0 / 2.4) - 0.055;
    return select(useLinear, v1, v2);
}

struct ComputeInput
{
    uint3 dispatchThreadID : SV_DispatchThreadID;
};

[shader("compute")]
[numthreads(8, 8, 1)]
func main(input: ComputeInput)->void
{
    let xy = int2(input.dispatchThreadID.xy);

    int width, height;
    inputTexture.GetDimensions(width, height);

    if (xy.x >= width || xy.y >= height)
    {
        return;
    }

    let color = inputTexture.Load( { xy, 0 }).rgb;
    let srgbColor = linearToSRGB(color);
    outputTexture[xy] = { srgbColor, 1.0 };
}
