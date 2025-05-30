import "../rtxns/Optimizers";

#define LOSS_SCALE 65536.0

struct PushConstants
{
    uint64_t seed;
    uint currentStep;
    float learningRate;
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
    uint encoderParamsSize;
    uint decoderParamsSize;
    uint textureSize;
    uint mollificationStep;
}

[[vk::binding(0)]]
uniform ConstantBuffer<UniformBuffer> uniforms;

[[vk::binding(1)]]
RWStructuredBuffer<half> encoderNetworkParams;

[[vk::binding(2)]]
RWStructuredBuffer<float> encoderNetworkParamsFloat;

[[vk::binding(3)]]
RWStructuredBuffer<half> encoderGradients;

[[vk::binding(4)]]
RWStructuredBuffer<float> encoderMoments1;

[[vk::binding(5)]]
RWStructuredBuffer<float> encoderMoments2;

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
    if (i >= uniforms.encoderParamsSize + uniforms.decoderParamsSize)
        return;

    if (i < uniforms.encoderParamsSize)
    {
        float encoderGradient = (float)encoderGradients[i];
        encoderGradients[i] = half(0.0);

        if (isfinite(encoderGradient))
        {
            float weightbias = encoderNetworkParamsFloat[i];

            optimizers::Adam optimizer = optimizers::Adam(encoderMoments1, encoderMoments2, pc.learningRate, LOSS_SCALE);

            float adjustedWeightbias = optimizer.step(weightbias, i, encoderGradient, pc.currentStep + 1);

            encoderNetworkParamsFloat[i] = adjustedWeightbias;
            encoderNetworkParams[i] = (half)adjustedWeightbias;
        }
        else
        {
            printf("Non-finite encoder gradient at index %u: %f\n", i, encoderGradient);
        }
    }
    else
    {
        i -= uniforms.encoderParamsSize;
        float decoderGradient = (float)decoderGradients[i];
        decoderGradients[i] = half(0.0);

        if (isfinite(decoderGradient))
        {
            float weightbias = decoderNetworkParamsFloat[i];

            optimizers::Adam optimizer = optimizers::Adam(decoderMoments1, decoderMoments2, pc.learningRate, LOSS_SCALE);

            float adjustedWeightbias = optimizer.step(weightbias, i, decoderGradient, pc.currentStep + 1);

            decoderNetworkParamsFloat[i] = adjustedWeightbias;
            decoderNetworkParams[i] = (half)adjustedWeightbias;
        }
        else
        {
            printf("Non-finite decoder gradient at index %u: %f\n", i, decoderGradient);
        }
    }
}
