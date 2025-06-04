import "../../rtxns/Activation";
import "../../rtxns/CooperativeVectorAutoDiff";
import "../../rtxns/CooperativeVectorDerivatives";
import "../../rtxns/CooperativeVectorFunctions";
import "../../rtxns/LinearOps";
import "../../rtxns/Loss";
import "../../rtxns/MLP";
import "../../rtxns/PCG32";
import "../../rtxns/Utils";

using namespace rtxns;
using namespace rtxns::mlp;

public struct Network
{
    public __init(
        StructuredBuffer<half> decoderMatrixBuffer,
        uint4 weightOffsets0,
        uint4 weightOffsets1,
        uint4 biasOffsets0,
        uint4 biasOffsets1)
    {
        decoderParameters = MatrixBiasBuffer<half>(decoderMatrixBuffer);
        weightOffsets = {
            weightOffsets0.x, weightOffsets0.y, weightOffsets0.z, weightOffsets0.w,
            weightOffsets1.x, weightOffsets1.y, weightOffsets1.z, weightOffsets1.w
        };
        biasOffsets = {
            biasOffsets0.x, biasOffsets0.y, biasOffsets0.z, biasOffsets0.w,
            biasOffsets1.x, biasOffsets1.y, biasOffsets1.z, biasOffsets1.w
        };
    }

    private static func transformComponentFunc(
        normal: float3,
        tangent: float3,
        v: float3)
        ->float3
    {
        let bitangent = cross(normal, tangent);
        let TBN = float3x3(tangent, bitangent, normal);
        return mul(TBN, v);
    }

    private static func transformFrameFunc(
        transformFrameInput: CoopVec<half, 12>,
        wo: float3,
        v1: float3,
        v2: float3,
        v3: float3,
        v4: float3)
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

    private static func combineLatentVectorAndTransformFrame(
        latentVector: CoopVec<half, 8>,
        transformFrameOutput: CoopVec<half, 30>)
        ->CoopVec<half, 38>
    {
        CoopVec<half, 38> result;
        for (var i = 0; i < 8; i++)
        {
            result[i] = latentVector[i];
        }
        for (var i = 0; i < 30; i++)
        {
            result[i + 8] = transformFrameOutput[i];
        }
        return result;
    }

    [mutating]
    public func forward(
        latentVector: CoopVec<half, 8>,
        wo: float3,
        v1: float3,
        v2: float3,
        v3: float3,
        v4: float3, )
        ->CoopVec<half, 3>
    {
        let act = rtxns::mlp::ReLUAct<half, 64>();
        let decoderFinalAct = rtxns::mlp::ExponentialAct<half, 3>();
        let matrixLayout = CoopVecMatrixLayout::InferencingOptimal;
        let componentType = CoopVecComponentType::Float16;

        using rtxns::mlp;

        CoopVec<half, 64> hidden;

        // decoder
        CoopVec<half, 3> output;

        // decoder transform frame
        CoopVec<half, 12> transformFrameInput;
        transformFrameInput = LinearOp<half, 12, 8>(
            latentVector,
            decoderParameters,
            uint2(weightOffsets[0], biasOffsets[0]),
            matrixLayout,
            componentType);
        transformFrameInput = tanh(transformFrameInput);
        CoopVec<half, 30> transformFrameOutput;
        transformFrameOutput = transformFrameFunc(
            transformFrameInput, wo, v1, v2, v3, v4);

        // decoder input
        CoopVec<half, 38> decoderInputParams;
        decoderInputParams = combineLatentVectorAndTransformFrame(
            latentVector,
            transformFrameOutput);

        // decoder 1st layer
        hidden = LinearOp<half, 64, 38>(
            decoderInputParams,
            decoderParameters,
            uint2(weightOffsets[1], biasOffsets[1]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 2nd layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            decoderParameters,
            uint2(weightOffsets[2], biasOffsets[2]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 3rd layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            decoderParameters,
            uint2(weightOffsets[3], biasOffsets[3]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 4th layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            decoderParameters,
            uint2(weightOffsets[4], biasOffsets[4]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 5th layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            decoderParameters,
            uint2(weightOffsets[5], biasOffsets[5]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 6th layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            decoderParameters,
            uint2(weightOffsets[6], biasOffsets[6]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 4th layer
        output = LinearOp<half, 3, 64>(
            hidden,
            decoderParameters,
            uint2(weightOffsets[7], biasOffsets[7]),
            matrixLayout,
            componentType);
        output = decoderFinalAct.eval(output - CoopVec<half, 3>(3.0, 3.0, 3.0));

        return output;
    }

    MatrixBiasBuffer<half> decoderParameters;
    uint[8] weightOffsets;
    uint[8] biasOffsets;
}
