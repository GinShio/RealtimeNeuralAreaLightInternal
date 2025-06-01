struct PushConstants
{
    uint64_t seed;
    uint currentStep;
}

[[push_constant]]
PushConstants pc;

struct UniformBuffer
{
    uint4 weightOffsets;
    uint4 biasOffsets;
    uint batchSize;
    uint paramsSize;
    float learningRate;
}

[[vk::binding(0)]]
uniform ConstantBuffer<UniformBuffer> uniforms;

[[vk::binding(1)]]
RWStructuredBuffer<half> networkParams;

[[vk::binding(2)]]
RWStructuredBuffer<float> networkParamsFloat;

[[vk::binding(3)]]
RWStructuredBuffer<half> gradients;

[[vk::binding(4)]]
RWStructuredBuffer<float> moments1;

[[vk::binding(5)]]
RWStructuredBuffer<float> moments2;

[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= uniforms.paramsSize)
        return;

    half param = networkParams[i];
    networkParamsFloat[i] = float(param);
    gradients[i] = half(0.0);
    moments1[i] = 0.0f;
    moments2[i] = 0.0f;
}
