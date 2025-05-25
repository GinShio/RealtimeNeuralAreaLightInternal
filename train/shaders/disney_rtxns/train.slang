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

#define INPUT_FEATURES 5

#define INPUT_NEURONS 30
#define OUTPUT_NEURONS 4

#define HIDDEN_NEURONS 32
#define NUM_HIDDEN_LAYERS 3

#define COMPONENT_WEIGHTS float4(1.0, 10.0, 1.0, 5.0)

#define NUM_TRANSITIONS (NUM_HIDDEN_LAYERS + 1)
#define NUM_TRANSITIONS_ALIGN4 ((NUM_TRANSITIONS + 3) / 4)

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

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint batchSize = uniforms.batchSize;
    uint i = tid.x;

    if (i >= batchSize)
        return;

    PCG32 rng = PCG32(pc.seed, i);

    // Using tangent coordinate system. N = (0,0,1)
    // L is arbitrary, but (N,L) >= 0 => L.z > 0, so generate random L in XZ plane's first quadrant
    float3 L;
    L.y = 0.f;
    sincos(rng.nextFloat() * PI / 2, L.z, L.x);

    // V is random direction, but (N,V) >= 0 => V.z > 0
    float sa, ca; // Azimuth [-PI/2,PI/2]
    sincos(-PI + 2 * PI * rng.nextFloat(), sa, ca);
    float se, ce; // Elevation [0,PI/2]
    sincos(PI / 2 * rng.nextFloat(), se, ce);
    float3 V = float3(ce * ca, ce * sa, se);

    float NdotL = L.z;
    float NdotV = V.z;

    float3 H = normalize(L + V);
    float NdotH = H.z;
    float LdotH = dot(L, H);

    float roughness = rng.nextFloat() * 0.7f + 0.3f;

    //----------- Calculate core shader part DIRECTLY
    float4 actualDisney = Disney(NdotL, NdotV, NdotH, LdotH, roughness);

    //----------- Training step
    float inputParams[5] = { NdotL, NdotV, NdotH, LdotH, roughness };
    CoopVec<half, 30> input = rtxns::EncodeFrequency<half, 5>(inputParams);

    uint[4] weightOffsets = {
        uniforms.weightOffsets[0], uniforms.weightOffsets[1],
        uniforms.weightOffsets[2], uniforms.weightOffsets[3]
    };
    uint[4] biasOffsets = {
        uniforms.biasOffsets[0], uniforms.biasOffsets[1],
        uniforms.biasOffsets[2], uniforms.biasOffsets[3]
    };
    var model = rtxns::mlp::TrainingMLP<half,
                                        NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS,
                                        CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
        networkParams,
        gradients,
        weightOffsets,
        biasOffsets);

    var hiddenActivation = rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>();
    var finalActivation = rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>();

    var output = model.forward(input, hiddenActivation, finalActivation);

    float4 predictedDisney = { output[0], output[1], output[2], output[3] };

    // calculate loss gradient
    CoopVec<half, 4> lossGradient;
    for (uint j = 0; j < 4; ++j)
    {
        let scale = LOSS_SCALE / (uniforms.batchSize * 4) / (output[j] * output[j] + 0.01);
        let deriv = 2.0 * (output[j] - actualDisney[j]);
        lossGradient[j] = half(scale * deriv * (COMPONENT_WEIGHTS[j]));
    }

    // call backward pass
    model.backward(input, hiddenActivation, finalActivation, HCoopVec<4>(lossGradient[0], lossGradient[1], lossGradient[2], lossGradient[3]));
}
