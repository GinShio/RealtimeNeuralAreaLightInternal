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

#define INPUT_NEURONS 5
#define OUTPUT_NEURONS 4

#define HIDDEN_NEURONS 64
#define NUM_HIDDEN_LAYERS 3

#define COMPONENT_WEIGHTS float4(1.f, 10.f, 1.f, 5.f)

#define NUM_TRANSITIONS (NUM_HIDDEN_LAYERS + 1)
#define NUM_TRANSITIONS_ALIGN4 ((NUM_TRANSITIONS + 3) / 4)

using namespace rtxns;
using namespace rtxns::mlp;

struct PushConstants {
    uint64_t seed;
    uint currentStep;
}

[[push_constant]]
PushConstants pc;

struct UniformBuffer {
    uint4 weightOffsets;
    uint4 biasOffsets;
    uint batchSize;
    uint paramsSize;
    float learningRate;
}

[[vk::binding(0)]]
uniform ConstantBuffer<UniformBuffer> uniforms;

[[vk::binding(1)]]
StructuredBuffer<half> networkParams;

[[vk::binding(2)]]
RWStructuredBuffer<half> gradients;

// [shader("compute")]
// [numthreads(64, 1, 1)]
// void main(uint3 dispatchThreadID : SV_DispatchThreadID)
// {
//     uint batchSize = uniforms.batchSize;
//     uint i = dispatchThreadID.x;

//     if (i >= batchSize)
//         return;

//     // PCG32 rng = PCG32(uniforms.seed, i);

//     // // Using tangent coordinate system. N = (0,0,1)
//     // // L is arbitrary, but (N,L) >= 0 => L.z > 0, so generate random L in XZ plane's first quadrant
//     // float3 L;
//     // L.y = 0.f;
//     // sincos(rng.nextFloat()*PI/2, L.z, L.x);

//     // // V is random direction, but (N,V) >= 0 => V.z > 0
//     // float sa, ca; // Azimuth [-PI/2,PI/2]
//     // sincos(-PI + 2 * PI * rng.nextFloat(), sa, ca);
//     // float se, ce; // Elevation [0,PI/2]
//     // sincos(PI/2 * rng.nextFloat(), se, ce);
//     // float3 V = float3(ce*ca, ce*sa, se);

//     // float NdotL = L.z;
//     // float NdotV = V.z;

//     // float3 H = normalize(L+V);
//     // float NdotH = H.z;
//     // float LdotH = dot(L,H);

//     // float roughness = rng.nextFloat()*0.7f+0.3f;

//     // //----------- Calculate core shader part DIRECTLY
//     // float4 actualDisney = Disney(NdotL, NdotV, NdotH, LdotH, roughness);
//     float4 actualDisney = Disney(1.0, 1.0, 1.0, 1.0, 1.0);

//     // //----------- Training step
//     // float params[INPUT_NEURONS] = {NdotL, NdotV, NdotH, LdotH, roughness};
//     // CoopVec<half, 5> inputParams = { params[0], params[1], params[2], params[3], params[4] };


//     // 入力ベクトル (5次元)
//     CoopVec<half, 5> inputParams;
//     for (uint j = 0; j < 5; ++j) {
//         inputParams[j] = half(1.0);
//     }

//     // var model = rtxns::mlp::TrainingMLP<half,
//     //     NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS,
//     //     CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
//     //     networkParams,
//     //     gradients,
//     //     rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>({uniforms.weightOffsets}),
//     //     rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>({uniforms.biasOffsets}));

//     uint[2] matrixOffsets = {uniforms.weightOffsets[0], uniforms.weightOffsets[1]};
//     uint[2] biasOffsets = {uniforms.biasOffsets[0], uniforms.biasOffsets[1]};

//     var model = rtxns::mlp::TrainingMLP<half,
//     1, 5, 64, 4,
//     CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
//     networkParams,
//     gradients,
//     matrixOffsets,
//     biasOffsets);

//     var hiddenActivation = rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>();
//     var finalActivation = rtxns::mlp::ExponentialAct<half, 4>();

//     // var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);

//     // float4 predictedDisney = { outputParams[0], outputParams[1], outputParams[2], outputParams[3] };

//     // float4 lossGradient = rtxns::mlp::L2Relative<float, 4>.deriv(actualDisney, predictedDisney, float4(LOSS_SCALE / (uniforms.batchSize * 4)) * COMPONENT_WEIGHTS);

//     // printf("Loss gradient for input %d: %f, %f, %f, %f\n", i, lossGradient[0], lossGradient[1], lossGradient[2], lossGradient[3]);

//     CoopVec<half, 4> gradOutput;
//     for (uint j = 0; j < 4; ++j) {
//         gradOutput[j] = half(1.0);
//     }

//     // model.backward(inputParams, hiddenActivation, finalActivation, rtxns::HCoopVec<OUTPUT_NEURONS>(lossGradient[0], lossGradient[1], lossGradient[2], lossGradient[3]));
//     model.backward(inputParams, hiddenActivation, finalActivation, gradOutput);

//     GroupMemoryBarrierWithGroupSync();

//     if (i == 0) {
//         printf("paramsSize = %u\n", uniforms.paramsSize);
//         printf("weightOffsets = %u, %u\n", matrixOffsets[0], matrixOffsets[1]);
//         printf("biasOffsets = %u, %u\n", biasOffsets[0], biasOffsets[1]);
//     }

//     if (i >= uniforms.paramsSize)
//         return;
//     if (gradients[i] != 0.0f) {
//         printf("gradient[%u] = %f\n", i, gradients[i]);
//     }
// }



[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint batchSize = uniforms.batchSize;
    uint i = tid.x;

    if (i >= batchSize)
        return;

    // 入力ベクトル (5次元)
    CoopVec<half, 5> input;
    for (uint j = 0; j < 5; ++j) {
        input[j] = half(0.5);
    }


    // データバッファ
    MatrixBiasBuffer<half> params = MatrixBiasBuffer<half>(networkParams);
    MatrixBiasBufferDifferential<half> diffs = MatrixBiasBufferDifferential<half>(gradients);

    DifferentialPair<CoopVec<half, 5>> inputPair = {
        input,
        CoopVec<half, 5>(half(0.0)) // 初期勾配は0
    };

    DifferentialPtrPair<MatrixBiasBuffer<half>> paramPair = {
        params,
        diffs
    };

    // 出力への勾配（∂L/∂y）を全て 0.1 とする
    CoopVec<half, 4> gradOutput;
    for (uint j = 0; j < 4; ++j) {
        gradOutput[j] = half(0.01);
    }

    // // 微分呼び出し
    // bwd_diff(LinearOp)(inputPair, paramPair, offsets,
    //     CoopVecMatrixLayout::TrainingOptimal,
    //     CoopVecComponentType::Float16, gradOutput);

    // coopVecReduceSumAccumulate(gradOutput, diffs.buffer, uniforms.biasOffsets[0]);

    // coopVecOuterProductAccumulate(gradOutput, input, diffs.buffer, matrixOffset, 0, CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16);
    // coopVecReduceSumAccumulate(gradOutput, diffs.buffer, biasOffset);

    // Activation関数の定義
    let act = ReLUAct<half, 64>();
    let finalAct = ExponentialAct<half, 4>();

    uint2[4] offsets = {
        uint2(uniforms.weightOffsets[0], uniforms.biasOffsets[0]),
        uint2(uniforms.weightOffsets[1], uniforms.biasOffsets[1]),
        uint2(uniforms.weightOffsets[2], uniforms.biasOffsets[2]),
        uint2(uniforms.weightOffsets[3], uniforms.biasOffsets[3])
    };

    // MLPモデルの呼び出し
    let output = model<ReLUAct<half, 64>, ExponentialAct<half, 4>>(
        input,
        params,
        offsets,
        ReLUAct<half, 64>(),
        ExponentialAct<half, 4>()
    );

    // 微分呼び出し
    bwd_diff(model)(inputPair, paramPair, offsets, act, finalAct, gradOutput);

    // if (pc.currentStep != 0)
    //     return;

    // if (i == 0) {
    //     printf("paramsSize = %u\n", uniforms.paramsSize);
    //     for (uint i = 0; i < 256; ++i)
    //     {
    //         printf("train phase gradient[%u] = %f\n", i, (float)gradients[i]);
    //     }
    // }
}


[Differentiable]
CoopVec<half, 4> model<Act : IActivation<half, 64>, FinalAct : IActivation<half, 4>>(
    CoopVec<half, 5> inputParams,
    MatrixBiasBuffer<half> parameters,
    uint2 layerOffsets[4],
    no_diff Act act,
    no_diff FinalAct finalAct
)
{
    let matrixLayout = CoopVecMatrixLayout::TrainingOptimal;
    let componentType = CoopVecComponentType::Float16;

    var params = act.eval(LinearOp<half, 64, 5>(inputParams, parameters, layerOffsets[0], matrixLayout, componentType));

    [ForceUnroll]
    for(int i = 1; i < 3; ++i)
        params = act.eval(LinearOp<half, 64, 64>(params, parameters, layerOffsets[i], matrixLayout, componentType));

    return finalAct.eval(LinearOp<half, 4, 64>(params, parameters, layerOffsets[3], matrixLayout, componentType));
}

// [Differentiable]
// static CoopVec<T, OUTPUTS> forward_s<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(
//     CoopVec<T, INPUTS> inputParams,
//     MatrixBiasBuffer<T> parameters,
//     uint2 layerOffsets[HIDDEN_LAYERS + 1],
//     no_diff Act act,
//     no_diff FinalAct finalAct
// )
// {
//     var params = act.eval(LinearOp<T, HIDDEN, INPUTS>(inputParams, parameters, layerOffsets[0], matrixLayout, componentType));

//     [ForceUnroll]
//     for(int i = 1; i < HIDDEN_LAYERS; ++i)
//         params = act.eval(LinearOp<T, HIDDEN, HIDDEN>(params, parameters, layerOffsets[i], matrixLayout, componentType));

//     return finalAct.eval(LinearOp<T, OUTPUTS, HIDDEN>(params, parameters, layerOffsets[HIDDEN_LAYERS], matrixLayout, componentType));
// }


// [numthreads(32, 1, 1)]
// void main(uint3 tid : SV_DispatchThreadID)
// {
//     uint i = tid.x;

//     // if (i >= uniforms.batchSize)
//     //     return;

//     // 書き込む位置
//     int biasOffset = uniforms.biasOffsets[0];

//     // 全スレッドが出力勾配 1.0 を持っていると仮定
//     CoopVec<half, 64> gradOutput;
//     for (uint j = 0; j < 64; ++j) {
//         gradOutput[j] = half(1.0);
//     }

//     if (i == 0) {
//         for (uint j = 0; j < 64; ++j)
//             printf("gradOutput[%u] = %f\n", j, gradOutput[j]);
//     }

//     GroupMemoryBarrierWithGroupSync();

//     // 各スレッドが呼ぶ（前提：スレッドごとに責任を持つユニットが割り当てられる想定）
//     coopVecReduceSumAccumulate(gradOutput, gradients, biasOffset);



//     GroupMemoryBarrierWithGroupSync();

//     if (i == 0) {
//         printf("paramsSize = %u\n", uniforms.paramsSize);
//     }

//     if (i >= uniforms.paramsSize)
//         return;
//     if (gradients[i] != 0.0f) {
//         printf("gradient[%u] = %f\n", i, gradients[i]);
//     }
// }




// // Helper macro to expand the register and space macros before concatenating tokens.
// // Declares a register space explicitly on DX12 and Vulkan, skips it on DX11.
// #ifdef TARGET_D3D11
// #define REGISTER_HELPER(TY,REG,SPACE) register(TY##REG)
// #else
// #define REGISTER_HELPER(TY,REG,SPACE) register(TY##REG, space##SPACE)
// #endif

// // Macros to declare bindings for various resource types in a cross-platform way
// // using register and space indices coming from other preprocessor macros.
// #define REGISTER_CBUFFER(reg,space) REGISTER_HELPER(b,reg,space)
// #define REGISTER_SAMPLER(reg,space) REGISTER_HELPER(s,reg,space)
// #define REGISTER_SRV(reg,space)     REGISTER_HELPER(t,reg,space)
// #define REGISTER_UAV(reg,space)     REGISTER_HELPER(u,reg,space)

// StructuredBuffer<half> gMLPParams               : REGISTER_SRV(1, 0);
// RWStructuredBuffer<half> gMLPParamsGradients    : REGISTER_UAV(2, 0);


// // [[vk::binding(1)]]
// // StructuredBuffer<half> gMLPParams;

// // [[vk::binding(2)]]
// // RWStructuredBuffer<half> gMLPParamsGradients ;

// [shader("compute")]
// [numthreads(64, 1, 1)]
// void main(uint3 dispatchThreadID : SV_DispatchThreadID)
// {
//     //----------- Randomly generate input parameters
//     uint idx = dispatchThreadID.x;
//     PCG32 rng = PCG32(uniforms.seed, idx);

//     // Using tangent coordinate system. N = (0,0,1)
//     // L is arbitrary, but (N,L) >= 0 => L.z > 0, so generate random L in XZ plane's first quadrant
//     float3 L;
//     L.y = 0.f;
//     sincos(rng.nextFloat()*PI/2, L.z, L.x);

//     // V is random direction, but (N,V) >= 0 => V.z > 0
//     float sa, ca; // Azimuth [-PI/2,PI/2]
//     sincos(-PI + 2 * PI * rng.nextFloat(), sa, ca);
//     float se, ce; // Elevation [0,PI/2]
//     sincos(PI/2 * rng.nextFloat(), se, ce);
//     float3 V = float3(ce*ca, ce*sa, se);

//     float NdotL = L.z;
//     float NdotV = V.z;

//     float3 H = normalize(L+V);
//     float NdotH = H.z;
//     float LdotH = dot(L,H);

//     // float roughness = rng.nextFloat()*0.7f+0.3f;
//     float roughness = rng.nextFloat();

//     //----------- Calculate core shader part DIRECTLY
//     float4 actualDisney = Disney(NdotL, NdotV, NdotH, LdotH, roughness);

//     //----------- Training step
//     float params[INPUT_FEATURES] = {NdotL, NdotV, NdotH, LdotH, roughness};
//     var inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

//     var model = rtxns::mlp::TrainingMLP<half,
//         NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS,
//         CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
//         gMLPParams,
//         gMLPParamsGradients,
//         rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>({uniforms.weightOffsets}),
//         rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>({uniforms.biasOffsets}));

//     var hiddenActivation = rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>();
//     var finalActivation = rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>();

//     var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);

//     float4 predictedDisney = { outputParams[0], outputParams[1], outputParams[2], outputParams[3] };

//     float4 lossGradient = rtxns::mlp::L2Relative<float, 4>.deriv(actualDisney, predictedDisney, float4(LOSS_SCALE / (uniforms.batchSize * 4)) * COMPONENT_WEIGHTS);

//     model.backward(inputParams, hiddenActivation, finalActivation, rtxns::HCoopVec<OUTPUT_NEURONS>(lossGradient[0], lossGradient[1], lossGradient[2], lossGradient[3]));


//     GroupMemoryBarrierWithGroupSync();

//     if (idx == 0) {
//         printf("paramsSize = %u\n", uniforms.paramsSize);
//     }

//     if (idx >= uniforms.paramsSize)
//         return;
//     if (gMLPParamsGradients[idx] != 0.0f) {
//         printf("gradient[%u] = %f\n", idx, gMLPParamsGradients[idx]);
//     }
// }
