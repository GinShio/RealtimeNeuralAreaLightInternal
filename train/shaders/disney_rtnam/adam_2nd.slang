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
    uint4 encoderWeightOffsets1;
    uint4 encoderWeightOffsets2;
    uint4 encoderBiasOffsets1;
    uint4 encoderBiasOffsets2;

    uint4 decoderWeightOffsets1;
    uint4 decoderWeightOffsets2;
    uint4 decoderBiasOffsets1;
    uint4 decoderBiasOffsets2;

    uint batchSize;
    float learningRate;
    uint latentTexturePixelSize;
    uint latentTextureParamsSize;
    uint decoderParamsSize;
    uint textureSize;
}

[[vk::binding(0)]]
uniform ConstantBuffer<UniformBuffer> uniforms;

[[vk::binding(1)]]
RWStructuredBuffer<half> latentTextureParams;

[[vk::binding(2)]]
RWStructuredBuffer<float> latentTextureParamsFloat;

[[vk::binding(3)]]
RWStructuredBuffer<float> latentTextureGradients;

[[vk::binding(4)]]
RWStructuredBuffer<float> latentTextureMoments1;

[[vk::binding(5)]]
RWStructuredBuffer<float> latentTextureMoments2;

[[vk::binding(6)]]
RWStructuredBuffer<half> decoderNetworkParams;

[[vk::binding(7)]]
RWStructuredBuffer<float> decoderNetworkParamsFloat;

[[vk::binding(8)]]
RWStructuredBuffer<half> decoderGradients;

[[vk::binding(9)]]
RWStructuredBuffer<float> decoderMoments1;

[[vk::binding(10)]]
RWStructuredBuffer<float> decoderMoments2;

[numthreads(32, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= uniforms.latentTextureParamsSize + uniforms.decoderParamsSize)
        return;

    if (i < uniforms.latentTextureParamsSize)
    {
        float latentTextureGradient = latentTextureGradients[i];
        // if (latentTextureGradient == 0.0)
        //     return;

        latentTextureGradients[i] = 0.0;

        if (isfinite(latentTextureGradient))
        {
            float weightbias = latentTextureParamsFloat[i];

            optimizers::Adam optimizer = optimizers::Adam(latentTextureMoments1, latentTextureMoments2, uniforms.learningRate, LOSS_SCALE);

            float adjustedWeightbias = optimizer.step(weightbias, i, latentTextureGradient, pc.currentStep + 1);

            latentTextureParamsFloat[i] = adjustedWeightbias;
            latentTextureParams[i] = (half)adjustedWeightbias;
        }
    }
    else
    {
        i -= uniforms.latentTextureParamsSize;
        float decoderGradient = (float)decoderGradients[i];
        decoderGradients[i] = half(0.0);

        if (isfinite(decoderGradient))
        {
            float weightbias = decoderNetworkParamsFloat[i];

            optimizers::Adam optimizer = optimizers::Adam(decoderMoments1, decoderMoments2, uniforms.learningRate, LOSS_SCALE);

            float adjustedWeightbias = optimizer.step(weightbias, i, decoderGradient, pc.currentStep + 1);

            decoderNetworkParamsFloat[i] = adjustedWeightbias;
            decoderNetworkParams[i] = (half)adjustedWeightbias;
        }
    }
}
