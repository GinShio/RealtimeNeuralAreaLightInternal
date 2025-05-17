[[vk::binding(0)]]
RWTexture2D<float4> inputTexture;
[[vk::binding(1)]]
RWTexture2D<float4> outputTexture;

func linearToSRGB(color: float3) -> float3
{
    let useLinear = color <= 0.0031308;
    let v1 = 12.92 * color;
    let v2 = 1.055 * pow(color, 1.0 / 2.4) - 0.055;
    return select(useLinear, v1, v2);
}

[shader("compute")]
[numthreads(8,8,1)]
func main(uint3 dispatchThreadID: SV_DispatchThreadID) -> void {
    let xy = int2(dispatchThreadID.xy);

    int width, height;
    inputTexture.GetDimensions(width, height);

    if (xy.x >= width || xy.y >= height) {
        return;
    }
    let color = inputTexture.Load(xy).rgb;
    let srgbColor = linearToSRGB(color);
    outputTexture[xy] = float4(srgbColor, 1.0);
}
