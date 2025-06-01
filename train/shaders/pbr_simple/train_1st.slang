import "../rtxns/Activation";
import "../rtxns/CooperativeVectorAutoDiff";
import "../rtxns/CooperativeVectorDerivatives";
import "../rtxns/CooperativeVectorFunctions";
import "../rtxns/LinearOps";
import "../rtxns/Loss";
import "../rtxns/MLP";
import "../rtxns/PCG32";
import "../rtxns/Utils";

import "coopvec_utils";

#define LOSS_SCALE 262144.0

using namespace rtxns;
using namespace rtxns::mlp;

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
StructuredBuffer<half> encoderNetworkParams;

[[vk::binding(2)]]
RWStructuredBuffer<half> encoderGradients;

[[vk::binding(3)]]
StructuredBuffer<half> decoderNetworkParams;

[[vk::binding(4)]]
RWStructuredBuffer<half> decoderGradients;

[[vk::binding(5)]]
StructuredBuffer<float> data;

struct Model
{
    __init(
        StructuredBuffer<half> encoderMatrixBuffer,
        RWStructuredBuffer<half> encoderDerivativeBuffer,
        StructuredBuffer<half> decoderMatrixBuffer,
        RWStructuredBuffer<half> decoderDerivativeBuffer)
    {
        encoderParameters = MatrixBiasBuffer<half>(encoderMatrixBuffer);
        encoderDerivatives = MatrixBiasBufferDifferential<half>(encoderDerivativeBuffer);
        decoderParameters = MatrixBiasBuffer<half>(decoderMatrixBuffer);
        decoderDerivatives = MatrixBiasBufferDifferential<half>(decoderDerivativeBuffer);
    }

    [Differentiable]
    private static func transformComponentFunc(
        normal: float3,
        tangent: float3,
        no_diff v: float3)
        ->float3
    {
        // normal = normalize(normal);
        // tangent = normalize(tangent - dot(tangent, normal) * normal);
        // let bitangent = normalize(cross(normal, tangent));
        let bitangent = cross(normal, tangent);
        let TBN = float3x3(tangent, bitangent, normal);
        return mul(TBN, v);
    }

    private static func transformFrameFunc(
        transformFrameInput: CoopVec<half, 12>,
        no_diff wo: float3,
        no_diff v1: float3,
        no_diff v2: float3,
        no_diff v3: float3,
        no_diff v4: float3)
        ->CoopVec<half, 30>
    {
        CoopVec<half, 30> result;
        for (var i = 0; i < 2; i++)
        {
            var normal = float3(
                transformFrameInput[i * 6 + 0],
                transformFrameInput[i * 6 + 1],
                transformFrameInput[i * 6 + 2]);
            var tangent = float3(
                transformFrameInput[i * 6 + 3],
                transformFrameInput[i * 6 + 4],
                transformFrameInput[i * 6 + 5]);

            let woTBN = transformComponentFunc(normal, tangent, wo);
            let v1TBN = transformComponentFunc(normal, tangent, v1);
            let v2TBN = transformComponentFunc(normal, tangent, v2);
            let v3TBN = transformComponentFunc(normal, tangent, v3);
            let v4TBN = transformComponentFunc(normal, tangent, v4);

            result[i * 15 + 0] = half(woTBN.x);
            result[i * 15 + 1] = half(woTBN.y);
            result[i * 15 + 2] = half(woTBN.z);
            result[i * 15 + 3] = half(v1TBN.x);
            result[i * 15 + 4] = half(v1TBN.y);
            result[i * 15 + 5] = half(v1TBN.z);
            result[i * 15 + 6] = half(v2TBN.x);
            result[i * 15 + 7] = half(v2TBN.y);
            result[i * 15 + 8] = half(v2TBN.z);
            result[i * 15 + 9] = half(v3TBN.x);
            result[i * 15 + 10] = half(v3TBN.y);
            result[i * 15 + 11] = half(v3TBN.z);
            result[i * 15 + 12] = half(v4TBN.x);
            result[i * 15 + 13] = half(v4TBN.y);
            result[i * 15 + 14] = half(v4TBN.z);
        }
        return result;
    }

    [BackwardDerivativeOf(transformFrameFunc)]
    private static func transformFrameFunc_Backward(
        inout transformFrameInput: DifferentialPair<CoopVec<half, 12>>,
        no_diff wo: float3,
        no_diff v1: float3,
        no_diff v2: float3,
        no_diff v3: float3,
        no_diff v4: float3,
        grad: CoopVec<half, 30>.Differential)
    {
        CoopVec<half, 12> transformFrameInputGrad;
        for (var i = 0; i < 2; i++)
        {
            var woTBNGrad = float3(
                grad[i * 15 + 0],
                grad[i * 15 + 1],
                grad[i * 15 + 2]);
            var v1TBNGrad = float3(
                grad[i * 15 + 3],
                grad[i * 15 + 4],
                grad[i * 15 + 5]);
            var v2TBNGrad = float3(
                grad[i * 15 + 6],
                grad[i * 15 + 7],
                grad[i * 15 + 8]);
            var v3TBNGrad = float3(
                grad[i * 15 + 9],
                grad[i * 15 + 10],
                grad[i * 15 + 11]);
            var v4TBNGrad = float3(
                grad[i * 15 + 12],
                grad[i * 15 + 13],
                grad[i * 15 + 14]);

            var normalP = float3(
                transformFrameInput.p[i * 6 + 0],
                transformFrameInput.p[i * 6 + 1],
                transformFrameInput.p[i * 6 + 2]);
            var normalD = float3(0.0);
            var normal = diffPair(normalP, normalD);

            var tangentP = float3(
                transformFrameInput.p[i * 6 + 3],
                transformFrameInput.p[i * 6 + 4],
                transformFrameInput.p[i * 6 + 5]);
            var tangentD = float3(0.0);
            var tangent = diffPair(tangentP, tangentD);

            bwd_diff(transformComponentFunc)(normal, tangent, wo, woTBNGrad);
            bwd_diff(transformComponentFunc)(normal, tangent, v1, v1TBNGrad);
            bwd_diff(transformComponentFunc)(normal, tangent, v2, v2TBNGrad);
            bwd_diff(transformComponentFunc)(normal, tangent, v3, v3TBNGrad);
            bwd_diff(transformComponentFunc)(normal, tangent, v4, v4TBNGrad);

            transformFrameInputGrad[i * 6 + 0] = half(normal.d.x);
            transformFrameInputGrad[i * 6 + 1] = half(normal.d.y);
            transformFrameInputGrad[i * 6 + 2] = half(normal.d.z);
            transformFrameInputGrad[i * 6 + 3] = half(tangent.d.x);
            transformFrameInputGrad[i * 6 + 4] = half(tangent.d.y);
            transformFrameInputGrad[i * 6 + 5] = half(tangent.d.z);
        }
        transformFrameInput = diffPair(
            transformFrameInput.p,
            transformFrameInputGrad);
    }

    private static func combineLatentVectorAndTransformFrame(
        latentVector: CoopVec<half, 8>,
        transformFrameOutput: CoopVec<half, 30>,
        no_diff v1Dist: float,
        no_diff v2Dist: float,
        no_diff v3Dist: float,
        no_diff v4Dist: float,
        no_diff area: float)
        ->CoopVec<half, 43>
    {
        CoopVec<half, 43> result;
        for (var i = 0; i < 8; i++)
        {
            result[i] = latentVector[i];
        }
        for (var i = 0; i < 30; i++)
        {
            result[i + 8] = transformFrameOutput[i];
        }
        result[38] = half(v1Dist);
        result[39] = half(v2Dist);
        result[40] = half(v3Dist);
        result[41] = half(v4Dist);
        result[42] = half(area);
        return result;
    }

    [BackwardDerivativeOf(combineLatentVectorAndTransformFrame)]
    private static func combineLatentVectorAndTransformFrame_Backward(
        inout latentVector: DifferentialPair<CoopVec<half, 8>>,
        inout transformFrameOutput: DifferentialPair<CoopVec<half, 30>>,
        no_diff v1Dist: float,
        no_diff v2Dist: float,
        no_diff v3Dist: float,
        no_diff v4Dist: float,
        no_diff area: float,
        grad: CoopVec<half, 43>.Differential)
    {
        CoopVec<half, 8> latentVectorGrad;
        for (var i = 0; i < 8; i++)
        {
            latentVectorGrad[i] = grad[i];
        }
        latentVector = diffPair(
            latentVector.p,
            latentVectorGrad);

        CoopVec<half, 30> transformFrameOutputGrad;
        for (var i = 0; i < 30; i++)
        {
            transformFrameOutputGrad[i] = grad[i + 8];
        }
        transformFrameOutput = diffPair(
            transformFrameOutput.p,
            transformFrameOutputGrad);
    }

    [Differentiable]
    static func forward_s(
        no_diff input: CoopVec<half, 8>,
        no_diff wo: float3,
        no_diff v1: float3,
        no_diff v2: float3,
        no_diff v3: float3,
        no_diff v4: float3,
        no_diff v1Dist: float,
        no_diff v2Dist: float,
        no_diff v3Dist: float,
        no_diff v4Dist: float,
        no_diff area: float,
        encoderParams: MatrixBiasBuffer<half>,
        decoderParams: MatrixBiasBuffer<half>, )
        ->CoopVec<half, 3>
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

        // decoder
        CoopVec<half, 3> output;

        let latentVectorTuple = duplicateCoopVec(latentVector);
        let latentVector_0 = latentVectorTuple._0;
        let latentVector_1 = latentVectorTuple._1;

        // decoder transform frame
        CoopVec<half, 12> transformFrameInput;
        transformFrameInput = LinearOp<half, 12, 8>(
            latentVector_0,
            decoderParams,
            uint2(
                uniforms.decoderWeightOffsets1[0],
                uniforms.decoderBiasOffsets1[0]),
            matrixLayout,
            componentType);
        CoopVec<half, 30> transformFrameOutput;
        transformFrameOutput = transformFrameFunc(
            transformFrameInput,
            wo,
            v1,
            v2,
            v3,
            v4);
        transformFrameOutput = tanh(transformFrameOutput);

        // decoder input
        CoopVec<half, 43> decoderInputParams;
        decoderInputParams = combineLatentVectorAndTransformFrame(
            latentVector_1,
            transformFrameOutput,
            v1Dist,
            v2Dist,
            v3Dist,
            v4Dist,
            area);

        // decoder 1st layer
        hidden = LinearOp<half, 64, 43>(
            decoderInputParams,
            decoderParams,
            uint2(
                uniforms.decoderWeightOffsets1[1],
                uniforms.decoderBiasOffsets1[1]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 2nd layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            decoderParams,
            uint2(
                uniforms.decoderWeightOffsets1[2],
                uniforms.decoderBiasOffsets1[2]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 3rd layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            decoderParams,
            uint2(
                uniforms.decoderWeightOffsets1[3],
                uniforms.decoderBiasOffsets1[3]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 4th layer
        output = LinearOp<half, 3, 64>(
            hidden,
            decoderParams,
            uint2(
                uniforms.decoderWeightOffsets2[0],
                uniforms.decoderBiasOffsets2[0]),
            matrixLayout,
            componentType);
        output = decoderFinalAct.eval(output);

        return output;
    }

    [mutating]
    func forward(
        input: CoopVec<half, 8>,
        wo: float3,
        no_diff v1: float3,
        no_diff v1Dist: float,
        no_diff v2: float3,
        no_diff v2Dist: float,
        no_diff v3: float3,
        no_diff v3Dist: float,
        no_diff v4: float3,
        no_diff v4Dist: float,
        no_diff area: float)
        ->CoopVec<half, 3>
    {
        return forward_s(
            input,
            wo,
            v1,
            v2,
            v3,
            v4,
            v1Dist,
            v2Dist,
            v3Dist,
            v4Dist,
            area,
            encoderParameters,
            decoderParameters);
    }

    [mutating]
    func backward(
        input: CoopVec<half, 8>,
        wo: float3,
        no_diff v1: float3,
        no_diff v1Dist: float,
        no_diff v2: float3,
        no_diff v2Dist: float,
        no_diff v3: float3,
        no_diff v3Dist: float,
        no_diff v4: float3,
        no_diff v4Dist: float,
        no_diff area: float,
        loss: CoopVec<half, 3>)
    {
        bwd_diff(forward_s)(
            input,
            wo,
            v1,
            v2,
            v3,
            v4,
            v1Dist,
            v2Dist,
            v3Dist,
            v4Dist,
            area,
            DifferentialPtrPair<MatrixBiasBuffer<half>>(encoderParameters, encoderDerivatives),
            DifferentialPtrPair<MatrixBiasBuffer<half>>(decoderParameters, decoderDerivatives),
            loss);
    }

    MatrixBiasBuffer<half> encoderParameters;
    MatrixBiasBufferDifferential<half> encoderDerivatives;
    MatrixBiasBuffer<half> decoderParameters;
    MatrixBiasBufferDifferential<half> decoderDerivatives;
}

[Differentiable]
func log1p1(x: float)->float
{
    var ret = x;
    ret = log(1.0 + ret);
    return ret;
}

[Differentiable]
func LogL1Loss(no_diff target: float3, predicted: float3, no_diff scale: float)->float3
{
    var loss = float3(0.0);
    for (var i = 0; i < 3; ++i)
    {
        let t = log1p1(target[i]);
        // let t = target[i];
        let p = predicted[i];
        loss[i] = abs(t - p);
    }
    return loss * scale;
}

func LogL1LossDerivative(
    target: float3,
    predicted: float3,
    scale: float,
    loss: float3)
    ->float3
{
    var predictedD = diffPair(predicted, float3(0.0));
    bwd_diff(LogL1Loss)(target, predictedD, scale, loss);
    return predictedD.d;
}

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint batchSize = uniforms.batchSize;
    uint i = tid.x;

    if (i >= batchSize)
        return;

    PCG32 rng = PCG32(pc.seed, i);

    // load train data
    let baseIndex = (i + uniforms.batchSize * pc.batchIndex) * 31;
    let baseColor = float3(
        data[baseIndex + 0],
        data[baseIndex + 1],
        data[baseIndex + 2]);
    let roughness = data[baseIndex + 3];
    let metallic = data[baseIndex + 4];
    let normal = float3(
        data[baseIndex + 5],
        data[baseIndex + 6],
        data[baseIndex + 7]);
    let wo = float3(
        data[baseIndex + 8],
        data[baseIndex + 9],
        data[baseIndex + 10]);
    let v1 = float3(
        data[baseIndex + 11],
        data[baseIndex + 12],
        data[baseIndex + 13]);
    let v1Dist = data[baseIndex + 14];
    let v2 = float3(
        data[baseIndex + 15],
        data[baseIndex + 16],
        data[baseIndex + 17]);
    let v2Dist = data[baseIndex + 18];
    let v3 = float3(
        data[baseIndex + 19],
        data[baseIndex + 20],
        data[baseIndex + 21]);
    let v3Dist = data[baseIndex + 22];
    let v4 = float3(
        data[baseIndex + 23],
        data[baseIndex + 24],
        data[baseIndex + 25]);
    let v4Dist = data[baseIndex + 26];
    let area = data[baseIndex + 27];
    let actualD = float3(
        data[baseIndex + 28],
        data[baseIndex + 29],
        data[baseIndex + 30]);

    CoopVec<half, 8> input = { baseColor.r, baseColor.g, baseColor.b, roughness, metallic, normal.x, normal.y, normal.z };

    var model = Model(
        encoderNetworkParams,
        encoderGradients,
        decoderNetworkParams,
        decoderGradients);

    CoopVec<half, 3> output = model.forward(
        input,
        wo,
        v1,
        v1Dist,
        v2,
        v2Dist,
        v3,
        v3Dist,
        v4,
        v4Dist,
        area);
    float3 predictedD = { output[0], output[1], output[2] };

    // calculate loss
    let scale = LOSS_SCALE / (uniforms.batchSize * 3);
    let loss = LogL1Loss(actualD, predictedD, scale) * float3(1.0, 1.0, 1.5);
    let lossD = LogL1LossDerivative(actualD, predictedD, scale, loss);

    // backward
    model.backward(
        input,
        wo,
        v1,
        v1Dist,
        v2,
        v2Dist,
        v3,
        v3Dist,
        v4,
        v4Dist,
        area,
        HCoopVec<3>(lossD[0], lossD[1], lossD[2]));
}
