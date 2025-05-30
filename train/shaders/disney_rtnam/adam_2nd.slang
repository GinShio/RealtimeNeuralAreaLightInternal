import "../rtxns/Optimizers";

#define LOSS_SCALE 65536.0

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
RWStructuredBuffer<float> latentTextureParams;

[[vk::binding(2)]]
RWStructuredBuffer<float> latentTextureGradients;

[[vk::binding(3)]]
StructuredBuffer<uint> latentTextureGradientIndex;

[[vk::binding(4)]]
RWStructuredBuffer<uint> latentTextureStepCount;

[[vk::binding(5)]]
RWStructuredBuffer<uint> latentTextureLock;

[[vk::binding(6)]]
RWStructuredBuffer<float> latentTextureMoments1;

[[vk::binding(7)]]
RWStructuredBuffer<float> latentTextureMoments2;

[[vk::binding(8)]]
RWStructuredBuffer<half> decoderNetworkParams;

[[vk::binding(9)]]
RWStructuredBuffer<float> decoderNetworkParamsFloat;

[[vk::binding(10)]]
RWStructuredBuffer<half> decoderGradients;

[[vk::binding(11)]]
RWStructuredBuffer<float> decoderMoments1;

[[vk::binding(12)]]
RWStructuredBuffer<float> decoderMoments2;

[numthreads(32, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= uniforms.decoderParamsSize + uniforms.batchSize)
        return;

    if (i < uniforms.decoderParamsSize)
    {
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
        else
        {
            printf("Non-finite decoder gradient at index %u: %f\n", i, decoderGradient);
        }
    }
    else
    {
        i -= uniforms.decoderParamsSize;

        for (var j = 0; j < 8; j++)
        {
            let index = latentTextureGradientIndex[i] * 8 + j;

            uint stepCount = 0;
            InterlockedAdd(latentTextureStepCount[index], 1, stepCount);

            float latentTextureGradient = latentTextureGradients[i * 8 + j];
            latentTextureGradients[i * 8 + j] = 0.0;

            if (isfinite(latentTextureGradient))
            {
                // lock for index
                uint outValue = 0;
                do
                {
                    InterlockedCompareExchange(latentTextureLock[index], 1, 0, outValue);
                } while (outValue != 0);

                float weightbias = latentTextureParams[index];

                optimizers::Adam optimizer = optimizers::Adam(latentTextureMoments1, latentTextureMoments2, uniforms.learningRate, LOSS_SCALE);

                float adjustedWeightbias = optimizer.step(weightbias, index, latentTextureGradient, stepCount + 1);

                latentTextureParams[index] = adjustedWeightbias;

                InterlockedExchange(latentTextureLock[index], 0); // release lock
            }
            else
            {
                printf("Non-finite latent texture gradient at index %u: %f\n", i, latentTextureGradient);
            }
        }
    }
}
