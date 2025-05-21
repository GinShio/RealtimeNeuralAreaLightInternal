[[vk::binding(0, 0)]]
Texture2D textures[];

[[vk::binding(0, 1)]]
SamplerState samplerList[];

struct VertexInput
{
  float3 position;
  float3 normal;
  float4 tangent;
  float2 uv;
}
