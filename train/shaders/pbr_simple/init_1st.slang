struct PushConstants
{
    uint64_t seed;
    uint currentStep;
    float learningRate;
    uint batchIndex;
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
        // Initialize encoder parameters
        half param = encoderNetworkParams[i];
        encoderNetworkParamsFloat[i] = float(param);
        encoderGradients[i] = half(0.0);
        encoderMoments1[i] = 0.0f;
        encoderMoments2[i] = 0.0f;
    }
    else
    {
        // Initialize decoder parameters
        i -= uniforms.encoderParamsSize;
        half param = decoderNetworkParams[i];
        decoderNetworkParamsFloat[i] = float(param);
        decoderGradients[i] = half(0.0);
        decoderMoments1[i] = 0.0f;
        decoderMoments2[i] = 0.0f;
    }
}
