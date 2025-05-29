import "../rtxns/Activation";
import "../rtxns/CooperativeVectorAutoDiff";
import "../rtxns/CooperativeVectorDerivatives";
import "../rtxns/CooperativeVectorFunctions";
import "../rtxns/LinearOps";
import "../rtxns/Loss";
import "../rtxns/MLP";
import "../rtxns/PCG32";
import "../rtxns/Utils";

import "disney";

#define LOSS_SCALE 128.0

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
StructuredBuffer<float> latentTextureParams;

[[vk::binding(2)]]
RWStructuredBuffer<float> latentTextureGradients;

[[vk::binding(3)]]
RWStructuredBuffer<uint> latentTextureGradientIndex;

[[vk::binding(4)]]
StructuredBuffer<half> decoderNetworkParams;

[[vk::binding(5)]]
RWStructuredBuffer<half> decoderGradients;

[[vk::binding(6)]]
Sampler2D baseColorSampler;

[[vk::binding(7)]]
Sampler2D roughnessMetallicSampler;

[[vk::binding(8)]]
Sampler2D normalSampler;

struct Model
{
    __init(
        StructuredBuffer<half> decoderMatrixBuffer,
        RWStructuredBuffer<half> decoderDerivativeBuffer)
    {
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
        no_diff wi: float3,
        no_diff wo: float3)
        ->CoopVec<half, 12>
    {
        CoopVec<half, 12> result;
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

            let wiTBN = transformComponentFunc(normal, tangent, wi);
            let woTBN = transformComponentFunc(normal, tangent, wo);

            result[i * 6 + 0] = half(wiTBN.x);
            result[i * 6 + 1] = half(wiTBN.y);
            result[i * 6 + 2] = half(wiTBN.z);
            result[i * 6 + 3] = half(woTBN.x);
            result[i * 6 + 4] = half(woTBN.y);
            result[i * 6 + 5] = half(woTBN.z);
        }
        return result;
    }

    [BackwardDerivativeOf(transformFrameFunc)]
    private static func transformFrameFunc_Backward(
        inout transformFrameInput: DifferentialPair<CoopVec<half, 12>>,
        no_diff wi: float3,
        no_diff wo: float3,
        grad: CoopVec<half, 12>.Differential)
    {
        CoopVec<half, 12> transformFrameInputGrad;
        for (var i = 0; i < 2; i++)
        {
            var wiTBNGrad = float3(
                grad[i * 6 + 0],
                grad[i * 6 + 1],
                grad[i * 6 + 2]);
            var woTBNGrad = float3(
                grad[i * 6 + 3],
                grad[i * 6 + 4],
                grad[i * 6 + 5]);

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

            bwd_diff(transformComponentFunc)(normal, tangent, wi, wiTBNGrad);
            bwd_diff(transformComponentFunc)(normal, tangent, wo, woTBNGrad);

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
        transformFrameOutput: CoopVec<half, 12>)
        ->CoopVec<half, 20>
    {
        CoopVec<half, 20> result;
        for (var i = 0; i < 8; i++)
        {
            result[i] = latentVector[i];
        }
        for (var i = 0; i < 12; i++)
        {
            result[i + 8] = transformFrameOutput[i];
        }
        return result;
    }

    [BackwardDerivativeOf(combineLatentVectorAndTransformFrame)]
    private static func combineLatentVectorAndTransformFrame_Backward(
        inout latentVector: DifferentialPair<CoopVec<half, 8>>,
        inout transformFrameOutput: DifferentialPair<CoopVec<half, 12>>,
        grad: CoopVec<half, 20>.Differential)
    {
        CoopVec<half, 8> latentVectorGrad;
        for (var i = 0; i < 8; i++)
        {
            latentVectorGrad[i] = grad[i];
        }
        latentVector = diffPair(
            latentVector.p,
            latentVectorGrad);

        CoopVec<half, 12> transformFrameOutputGrad;
        for (var i = 0; i < 12; i++)
        {
            transformFrameOutputGrad[i] = grad[i + 8];
        }
        transformFrameOutput = diffPair(
            transformFrameOutput.p,
            transformFrameOutputGrad);
    }

    [Differentiable]
    static func forward_s(
        latentVector: CoopVec<half, 8>,
        no_diff wi: float3,
        no_diff wo: float3,
        decoderParams: MatrixBiasBuffer<half>, )
        ->CoopVec<half, 3>
    {
        let act = rtxns::mlp::ReLUAct<half, 64>();
        let encoderFinalAct = rtxns::mlp::SigmoidAct<half, 8>();
        let decoderFinalAct = rtxns::mlp::ExponentialAct<half, 3>();
        let matrixLayout = CoopVecMatrixLayout::TrainingOptimal;
        let componentType = CoopVecComponentType::Float16;

        using rtxns::mlp;

        CoopVec<half, 64> hidden;

        // decoder
        CoopVec<half, 3> output;

        // decoder transform frame
        CoopVec<half, 12> transformFrameInput;
        transformFrameInput = LinearOp<half, 12, 8>(
            latentVector,
            decoderParams,
            uint2(
                uniforms.decoderWeightOffsets1[0],
                uniforms.decoderBiasOffsets1[0]),
            matrixLayout,
            componentType);
        CoopVec<half, 12> transformFrameOutput;
        transformFrameOutput = transformFrameFunc(
            transformFrameInput,
            wi,
            wo);

        // decoder input
        CoopVec<half, 20> decoderInputParams;
        decoderInputParams = combineLatentVectorAndTransformFrame(
            latentVector,
            transformFrameOutput);

        // decoder 1st layer
        hidden = LinearOp<half, 64, 20>(
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
        latentVector: CoopVec<half, 8>,
        wi: float3,
        wo: float3)
        ->CoopVec<half, 3>
    {
        return forward_s(
            latentVector,
            wi,
            wo,
            decoderParameters);
    }

    [mutating]
    func backward(
        inout latentVector: DifferentialPair<CoopVec<half, 8>>,
        wi: float3,
        wo: float3,
        loss: CoopVec<half, 3>)
    {
        bwd_diff(forward_s)(
            latentVector,
            wi,
            wo,
            DifferentialPtrPair<MatrixBiasBuffer<half>>(decoderParameters, decoderDerivatives),
            loss);
    }

    MatrixBiasBuffer<half> decoderParameters;
    MatrixBiasBufferDifferential<half> decoderDerivatives;
}

func localToTangent(v: float3)->float3x3
{
    let up = abs(v.z) < 0.9999 ? float3(0, 0, 1) : float3(1, 0, 0);
    let tangent = normalize(cross(up, v));
    let bitangent = cross(v, tangent);
    return float3x3(tangent, bitangent, v);
}

func sampleWiWo(
    inout rng: PCG32,
    out wi: float3,
    out wo: float3)
{
    let thetaH = acos(sqrt(1.0 - rng.nextFloat()));
    let phiH = 2.0 * PI * rng.nextFloat();
    let H = float3(
        sin(thetaH) * cos(phiH),
        sin(thetaH) * sin(phiH),
        cos(thetaH));

    let thetaDelta = acos(sqrt(1.0 - rng.nextFloat()));
    let phiDelta = 2.0 * PI * rng.nextFloat();
    let delta = float3(
        sin(thetaDelta) * cos(phiDelta),
        sin(thetaDelta) * sin(phiDelta),
        cos(thetaDelta));

    let M = localToTangent(H);
    let d = mul(M, delta);

    wi = normalize(d);
    wo = 2.0 * dot(wi, H) * H - wi;
}

func computeMipCount(width: int)->int
{
    return int(floor(log2(float(width)))) + 1;
}

func sampleMipLevel(u: float, mipCount: int)->float
{
    let bias = log(2.0);
    let l_max = float(mipCount - 1);
    let l = -log(1.0 - u) / bias;
    return clamp(l, 0.0, l_max);
}

func srgbToLinear(color: float3)->float3
{
    let useLinear = color <= 0.04045;
    let v1 = color / 12.92;
    let v2 = pow((color + 0.055) / 1.055, 2.4);
    return select(useLinear, v1, v2);
}

func createTbn(v: float3, normal: float3)
    ->float3x3
{
    var T = float3(1, 0, 0);
    var N = normalize(normal);
    T = normalize(T - dot(T, N) * N);
    let B = normalize(cross(N, T));
    return float3x3(T, B, N);
}

func actualBrdfFunc(wi: float3, wo: float3, uv: float2, mipLevel: float)->float3
{
    // TODO filtering and mollification

    let baseColor = baseColorSampler.SampleLevel(uv, mipLevel);
    let metallicRoughness = roughnessMetallicSampler.SampleLevel(uv, mipLevel);
    let normal = normalSampler.SampleLevel(uv, mipLevel).xyz * 2 - 1;
    let roughness = metallicRoughness.g;
    let metallic = metallicRoughness.b;

    let TBN = createTbn(wo, normal);

    let wi_tangent = mul(TBN, wi);
    let wo_tangent = mul(TBN, wo);
    return DisneyBRDF(wi_tangent, wo_tangent, srgbToLinear(baseColor.rgb), roughness, metallic);
}

func inputParametersIndex(uv: float2, mipLevel: float)->uint
{
    // Select lower mip level
    let mipLevelFloor = floor(mipLevel);

    // Calculate pixel offset in latent texture buffer
    var currentMipLevel = 0;
    var currentMipSize = uniforms.textureSize;
    var currentPixelOffset = 0;
    while (currentMipSize > 1)
    {
        if (mipLevelFloor == currentMipLevel)
            break;

        currentPixelOffset += currentMipSize * currentMipSize;
        currentMipSize /= 2;
        currentMipLevel++;
    }
    let st = uint2(
        uint(uv.x * currentMipSize),
        uint(uv.y * currentMipSize));
    let latentTexturePixelIndex = currentPixelOffset + st.y * currentMipSize + st.x;

    if (latentTexturePixelIndex >= uniforms.latentTexturePixelSize)
    {
        printf("latentTexturePixelIndex out of bounds: mip level %d, st.x %d, st.y %d\n", currentMipLevel, st.x, st.y);
    }

    // Calculate index in latent texture parameters buffer
    return latentTexturePixelIndex * 8;
}

[Differentiable]
func log1p4(x: float)->float
{
    var ret = x;
    ret = log(1.0 + ret);
    ret = log(1.0 + ret);
    ret = log(1.0 + ret);
    ret = log(1.0 + ret);
    return ret;
}

[Differentiable]
func LogL1Loss(no_diff target: float3, predicted: float3, no_diff scale: float)->float
{
    var loss : float = 0.0;
    for (var i = 0; i < 3; ++i)
    {
        let t = log1p4(target[i]);
        let p = log1p4(predicted[i]);
        loss += abs(t - p);
    }
    return loss * scale;
}

func LogL1LossDerivative(
    target: float3,
    predicted: float3,
    scale: float,
    loss: float)
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

    // sample wi and wo
    float3 wi;
    float3 wo;
    sampleWiWo(rng, wi, wo);

    // sample uv
    let uv = float2(rng.nextFloat(), rng.nextFloat());
    let mipLevel = sampleMipLevel(rng.nextFloat(), computeMipCount(uniforms.textureSize));

    // calculate actual BRDF
    let actualBrdf = actualBrdfFunc(wi, wo, uv, mipLevel);

    // calculate model BRDF
    let parameterIndex = inputParametersIndex(uv, mipLevel);
    CoopVec<half, 8> input = {
        half(latentTextureParams[parameterIndex + 0]),
        half(latentTextureParams[parameterIndex + 1]),
        half(latentTextureParams[parameterIndex + 2]),
        half(latentTextureParams[parameterIndex + 3]),
        half(latentTextureParams[parameterIndex + 4]),
        half(latentTextureParams[parameterIndex + 5]),
        half(latentTextureParams[parameterIndex + 6]),
        half(latentTextureParams[parameterIndex + 7]),
    };

    var model = Model(
        decoderNetworkParams,
        decoderGradients);

    CoopVec<half, 3> output = model.forward(input, wi, wo);
    float3 predictedBrdf = { output[0], output[1], output[2] };

    // calculate loss
    let scale = LOSS_SCALE / (uniforms.batchSize * 3);
    let loss = LogL1Loss(actualBrdf, predictedBrdf, scale);
    let lossD = LogL1LossDerivative(actualBrdf, predictedBrdf, scale, loss);

    // backward
    CoopVec<half, 8> inputDerivative;
    var inputParams = diffPair(input, inputDerivative);
    model.backward(inputParams, wi, wo, HCoopVec<3>(lossD[0], lossD[1], lossD[2]));

    // store gradients
    latentTextureGradients[i * 8 + 0] += float(inputParams.d[0]);
    latentTextureGradients[i * 8 + 1] += float(inputParams.d[1]);
    latentTextureGradients[i * 8 + 2] += float(inputParams.d[2]);
    latentTextureGradients[i * 8 + 3] += float(inputParams.d[3]);
    latentTextureGradients[i * 8 + 4] += float(inputParams.d[4]);
    latentTextureGradients[i * 8 + 5] += float(inputParams.d[5]);
    latentTextureGradients[i * 8 + 6] += float(inputParams.d[6]);
    latentTextureGradients[i * 8 + 7] += float(inputParams.d[7]);
    latentTextureGradientIndex[i] = parameterIndex;
}
