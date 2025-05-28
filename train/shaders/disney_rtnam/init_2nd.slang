import "../rtxns/Activation";
import "../rtxns/CooperativeVectorAutoDiff";
import "../rtxns/CooperativeVectorDerivatives";
import "../rtxns/CooperativeVectorFunctions";
import "../rtxns/LinearOps";
import "../rtxns/Loss";
import "../rtxns/MLP";
import "../rtxns/PCG32";
import "../rtxns/Utils";

using namespace rtxns;
using namespace rtxns::mlp;

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

[[vk::binding(11)]]
StructuredBuffer<half> encoderNetworkParams;

[[vk::binding(12)]]
Sampler2D baseColorSampler;

[[vk::binding(13)]]
Sampler2D roughnessMetallicSampler;

[[vk::binding(14)]]
Sampler2D normalSampler;

struct EncoderModel {
    __init(StructuredBuffer<half> encoderMatrixBuffer) {
        encoderParameters = MatrixBiasBuffer<half>(encoderMatrixBuffer);
    }

    static func forward_s(
        no_diff input: CoopVec<half, 8>,
        encoderParams: MatrixBiasBuffer<half>,
    )->CoopVec<half, 8>
    {
        let act = rtxns::mlp::ReLUAct<half, 64>();
        let encoderFinalAct = rtxns::mlp::SigmoidAct<half, 8>();
        let decoderFinalAct = rtxns::mlp::ExponentialAct<half, 3>();
        let matrixLayout = CoopVecMatrixLayout::TrainingOptimal;
        let componentType = CoopVecComponentType::Float16;

        using rtxns::mlp;

        // encoder
        CoopVec<half, 64> hidden;
        CoopVec<half, 8> latentVector;

        // encoder 1st layer
        hidden = LinearOp<half, 64, 8>(
            input,
            encoderParams,
            uint2(
                uniforms.encoderWeightOffsets1[0],
                uniforms.encoderBiasOffsets1[0]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // encoder 2nd layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            encoderParams,
            uint2(
                uniforms.encoderWeightOffsets1[1],
                uniforms.encoderBiasOffsets1[1]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // encoder 3rd layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            encoderParams,
            uint2(
                uniforms.encoderWeightOffsets1[2],
                uniforms.encoderBiasOffsets1[2]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // encoder 4th layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            encoderParams,
            uint2(
                uniforms.encoderWeightOffsets1[3],
                uniforms.encoderBiasOffsets1[3]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // encoder 5th layer
        latentVector = LinearOp<half, 8, 64>(
            hidden,
            encoderParams,
            uint2(
                uniforms.encoderWeightOffsets2[0],
                uniforms.encoderBiasOffsets2[0]),
            matrixLayout,
            componentType);
        latentVector = encoderFinalAct.eval(latentVector);

        return latentVector;
    }

    [mutating]
    func forward(input: CoopVec<half, 8>) -> CoopVec<half, 8>
    {
        return forward_s(input, encoderParameters);
    }

    MatrixBiasBuffer<half> encoderParameters;
}

[numthreads(32, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= uniforms.latentTexturePixelSize + uniforms.decoderParamsSize)
        return;

    if (i < uniforms.latentTexturePixelSize)
    {
        // Initialize latent texture parameters

        // calculate mip level and pixel offset
        var mipLevel = 0;
        var currentMipSize = uniforms.textureSize;
        var currentPixelOffset = 0;
        while (currentMipSize > 1)
        {
            if (i < currentPixelOffset + currentMipSize * currentMipSize)
                break;

            currentPixelOffset += currentMipSize * currentMipSize;
            currentMipSize /= 2;
            mipLevel++;
        }

        // calculate UV
        let uv = float2(
            (i - currentPixelOffset) % currentMipSize,
            (i - currentPixelOffset) / currentMipSize);

        // Sample parameters from textures
        let baseColor = baseColorSampler.SampleLevel(uv, mipLevel);
        let roughnessMetallic = roughnessMetallicSampler.SampleLevel(uv, mipLevel);
        let normal = normalSampler.SampleLevel(uv, mipLevel);

        // execute encoder
        CoopVec<half, 8> input = {
            half(baseColor.r),
            half(baseColor.g),
            half(baseColor.b),
            half(roughnessMetallic.r),
            half(roughnessMetallic.g),
            half(roughnessMetallic.b),
            half(normal.r),
            half(normal.g)
        };
        var model = EncoderModel(encoderNetworkParams);
        CoopVec<half, 8> param = model.forward(input);

        // Init parameters
        for (uint j = 0; j < 8; j++)
        {
            latentTextureParams[i * 8 + j] = input[j];
            latentTextureParamsFloat[i * 8 + j] = float(input[j]);
            latentTextureGradients[i * 8 + j] = 0.0;
            latentTextureMoments1[i * 8 + j] = 0.0f;
            latentTextureMoments2[i * 8 + j] = 0.0f;
        }
    }
    else
    {
        // Initialize decoder parameters
        i -= uniforms.latentTexturePixelSize;
        half param = decoderNetworkParams[i];
        decoderNetworkParamsFloat[i] = float(param);
        decoderGradients[i] = half(0.0);
        decoderMoments1[i] = 0.0f;
        decoderMoments2[i] = 0.0f;
    }
}
