[[vk::binding(0, 0)]]
RWTexture2D<float4> input_image;
[[vk::binding(1, 0)]]
RWTexture2D<float4> output_image;

struct MipmapPushConstant
{
    uint src_mip_level;
    uint dst_mip_level;
    uint src_width;
    uint src_height;
};

[[push_constant]]
MipmapPushConstant pc;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 dst_coord = dispatchThreadId.xy;
    uint dst_width = max(pc.src_width >> 1, 1);
    uint dst_height = max(pc.src_height >> 1, 1);

    if (dst_coord.x >= dst_width || dst_coord.y >= dst_height)
        return;

    uint2 src_coord0 = dst_coord * 2;
    uint2 src_coord1 = src_coord0 + uint2(1, 0);
    uint2 src_coord2 = src_coord0 + uint2(0, 1);
    uint2 src_coord3 = src_coord0 + uint2(1, 1);

    float4 c0 = input_image[src_coord0];
    float4 c1 = float4(0, 0, 0, 0);
    float4 c2 = float4(0, 0, 0, 0);
    float4 c3 = float4(0, 0, 0, 0);

    if (src_coord1.x < pc.src_width)
        c1 = input_image[src_coord1];
    if (src_coord2.y < pc.src_height)
        c2 = input_image[src_coord2];
    if (src_coord1.x < pc.src_width && src_coord2.y < pc.src_height)
        c3 = input_image[src_coord3];

    float4 avg = (c0 + c1 + c2 + c3) / 4.0;

    output_image[dst_coord] = avg;
}
