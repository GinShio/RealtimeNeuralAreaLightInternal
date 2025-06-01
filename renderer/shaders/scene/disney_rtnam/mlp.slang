import "../base.slang";

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

[[vk::binding(0, 2)]]
StructuredBuffer<half> params;

struct LightParams
{
    float3 position;
    float intensity;
    float3 color;
    uint padding;
}

struct Uniforms
{
    uint4 weightOffsets0;
    uint4 weightOffsets1;
    uint4 biasOffsets0;
    uint4 biasOffsets1;

    float4x4 view;
    float4x4 projection;

    uint samplerIndex;
    uint latentTexture0Index;
    uint latentTexture1Index;

    uint light_count;
    LightParams light[3];
}

[[vk::binding(0, 3)]]
uniform ConstantBuffer<Uniforms> uniforms;

struct PushConstants
{
    float4x4 model;
    uint baseColorIndex;
    uint normalIndex;
    uint metallicRoughnessIndex;
    uint padding;
    float3 baseColorOverrideValue;
    uint baseColorOverride;
    float roughnessOverrideValue;
    uint roughnessOverride;
    float metallicOverrideValue;
    uint metallicOverride;
}

[[push_constant]]
PushConstants pc;

struct VertexOutput
{
    float4 position : SV_POSITION;
    float3 viewPosition : TEXCOORD0;
    float3 viewNormal : TEXCOORD1;
    float4 viewTangent : TEXCOORD2;
    float2 uv : TEXCOORD3;
}

float3x3 inverse3x3(float3x3 m)
{
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

    float det = a00 * (a11 * a22 - a12 * a21) -
                a01 * (a10 * a22 - a12 * a20) +
                a02 * (a10 * a21 - a11 * a20);

    if (abs(det) < 1e-6)
        return float3x3(0.0);

    float invDet = 1.0 / det;

    return float3x3(
        +(a11 * a22 - a12 * a21) * invDet,
        -(a01 * a22 - a02 * a21) * invDet,
        +(a01 * a12 - a02 * a11) * invDet,

        -(a10 * a22 - a12 * a20) * invDet,
        +(a00 * a22 - a02 * a20) * invDet,
        -(a00 * a12 - a02 * a10) * invDet,

        +(a10 * a21 - a11 * a20) * invDet,
        -(a00 * a21 - a01 * a20) * invDet,
        +(a00 * a11 - a01 * a10) * invDet);
}

func srgbToLinear(color: float3)->float3
{
    let useLinear = color <= 0.04045;
    let v1 = color / 12.92;
    let v2 = pow((color + 0.055) / 1.055, 2.4);
    return select(useLinear, v1, v2);
}

[shader("vertex")]
func vsMain(input: VertexInput)->VertexOutput
{
    let viewModel = mul(uniforms.view, pc.model);
    let viewModelIT = inverse3x3(transpose(float3x3(viewModel)));

    let viewPosition = mul(viewModel, float4(input.position, 1.0));

    let viewNormal = normalize(mul(viewModelIT, input.normal));

    let viewTangent = normalize(mul(viewModelIT, input.tangent.xyz));
    let viewTangentW = input.tangent.w;

    let uv = input.uv;

    let position = mul(uniforms.projection, viewPosition);

    return {
        position,
        viewPosition.xyz,
        viewNormal.xyz,
        float4(viewTangent.xyz, viewTangentW),
        uv,
    };
}

struct FragmentOutput
{
    float4 color : SV_TARGET;
}

struct Model
{
    __init(StructuredBuffer<half> decoderMatrixBuffer)
    {
        decoderParameters = MatrixBiasBuffer<half>(decoderMatrixBuffer);
    }

    private static func transformComponentFunc(
        normal: float3,
        tangent: float3,
        no_diff v: float3)
        ->float3
    {
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

    static func forward_s(
        latentVector: CoopVec<half, 8>,
        no_diff wi: float3,
        no_diff wo: float3,
        decoderParams: MatrixBiasBuffer<half>, )
        ->CoopVec<half, 3>
    {
        let act = rtxns::mlp::ReLUAct<half, 64>();
        let encoderFinalAct = rtxns::mlp::SigmoidAct<half, 8>();
        let decoderFinalAct = rtxns::mlp::SigmoidAct<half, 3>();
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
            decoderParams,
            uint2(
                uniforms.weightOffsets0[0],
                uniforms.biasOffsets0[0]),
            matrixLayout,
            componentType);
        CoopVec<half, 12> transformFrameOutput;
        transformFrameOutput = transformFrameFunc(
            transformFrameInput,
            wi,
            wo);
        transformFrameOutput = tanh(transformFrameOutput);

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
                uniforms.weightOffsets0[1],
                uniforms.biasOffsets0[1]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 2nd layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            decoderParams,
            uint2(
                uniforms.weightOffsets0[2],
                uniforms.biasOffsets0[2]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 3rd layer
        hidden = LinearOp<half, 64, 64>(
            hidden,
            decoderParams,
            uint2(
                uniforms.weightOffsets0[3],
                uniforms.biasOffsets0[3]),
            matrixLayout,
            componentType);
        hidden = act.eval(hidden);

        // decoder 4th layer
        output = LinearOp<half, 3, 64>(
            hidden,
            decoderParams,
            uint2(
                uniforms.weightOffsets1[0],
                uniforms.biasOffsets1[0]),
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

    MatrixBiasBuffer<half> decoderParameters;
}

func inverse_log1p4(y: float3)->float3
{
    float3 ret = y;
    ret = exp(ret) - 1.0;
    ret = exp(ret) - 1.0;
    ret = exp(ret) - 1.0;
    ret = exp(ret) - 1.0;
    return ret;
}

[shader("fragment")]
func fsMain(input: VertexOutput)->FragmentOutput
{
    let sampler = samplerList[uniforms.samplerIndex];

    let latentValue0 = textures[uniforms.latentTexture0Index].Sample(sampler, input.uv);
    let latentValue1 = textures[uniforms.latentTexture1Index].Sample(sampler, input.uv);
    CoopVec<half, 8> latentVector = {
        latentValue0.r,
        latentValue0.g,
        latentValue0.b,
        latentValue0.a,
        latentValue1.r,
        latentValue1.g,
        latentValue1.b,
        latentValue1.a,
    };

    var model = Model(params);

    let normal = normalize(input.viewNormal);
    let tangent = normalize(input.viewTangent.xyz);

    let N = normalize(normal);
    let T = normalize(tangent - dot(tangent, N) * N);
    let B = cross(N, T) * input.viewTangent.w;
    let TBN = float3x3(T, B, N);

    let Vt = mul(TBN, normalize(-input.viewPosition));
    let Nt = float3(0, 0, 1);

    var color = float3(0.0, 0.0, 0.0);
    for (var i = 0; i < uniforms.light_count; i++)
    {
        let light = uniforms.light[i];

        let lightViewPosition = mul(uniforms.view, float4(light.position, 1.0)).xyz;
        let Lt = mul(TBN, normalize(lightViewPosition - input.viewPosition));

        if (dot(Lt, Nt) < 0.0 || dot(Vt, Nt) < 0.0)
            continue;

        let output = model.forward(latentVector, normalize(Lt), normalize(Vt));
        let brdf = inverse_log1p4(float3(output[0], output[1], output[2]));
        let lightColor = srgbToLinear(light.color);
        let cosTheta = max(dot(Lt, Nt), 0.0);
        color += brdf * lightColor * light.intensity * cosTheta;
    }

    return {
        { color, 1.0 },
    };
}
