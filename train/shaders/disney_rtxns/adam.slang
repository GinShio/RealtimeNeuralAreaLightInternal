import "../rtxns/Optimizers";

#define LOSS_SCALE 128.0

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
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= uniforms.paramsSize)
        return;

    float gradient = (float)gradients[i];
    gradients[i] = half(0.0);

    if (isfinite(gradient))
    {
        float weightbias = networkParamsFloat[i];

        optimizers::Adam optimizer = optimizers::Adam(moments1, moments2, uniforms.learningRate, LOSS_SCALE);

        float adjustedWeightbias = optimizer.step(weightbias, i, gradient, pc.currentStep + 1);

        networkParamsFloat[i] = adjustedWeightbias;
        networkParams[i] = (half)adjustedWeightbias;
    }
}
