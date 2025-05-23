import "base.slang";

import "../rtxns/CooperativeVectorFunctions";
import "../rtxns/Utils";
import "../rtxns/LinearOps";

[[vk::binding(0, 2)]]
StructuredBuffer<half> params;

struct PushConstants {
  float3 fillColor;
  uint padding0;
  uint weight_offsets[3];
  uint padding1;
  uint bias_offsets[3];
  uint padding2;
}

[[push_constant]]
PushConstants pc;

struct VertexOutput
{
  float4 position : SV_POSITION;
  float3 normal : NORMAL;
  float4 tangent : TANGENT;
  float2 uv : TEXCOORD;
}

func srgbToLinear(color: float3) -> float3
{
  let useLinear = color <= 0.04045;
  let v1 = color / 12.92;
  let v2 = pow((color + 0.055) / 1.055, 2.4);
  return select(useLinear, v1, v2);
}

[shader("vertex")]
func vsMain(input: VertexInput) -> VertexOutput
{
  return {
    { input.position, 1.0 },
    input.normal,
    input.tangent,
    input.uv,
  };
}

struct FragmentOutput
{
  float4 color : SV_TARGET;
}

[shader("fragment")]
func fsMain(input: VertexOutput) -> FragmentOutput
{
  CoopVec<half, 3> inputParams;
  CoopVec<half, 32> hiddenParams;
  CoopVec<half, 3> outputParams;

  inputParams = { pc.fillColor.x, pc.fillColor.y, pc.fillColor.z };

  hiddenParams = rtxns::LinearOp<half, 32, 3>(
    inputParams,
    params,
    pc.weight_offsets[0],
    pc.bias_offsets[0],
    CoopVecMatrixLayout::InferencingOptimal,
    CoopVecComponentType::Float16
  );
  hiddenParams = rtxns::relu(hiddenParams);

  hiddenParams = rtxns::LinearOp<half, 32, 32>(
    hiddenParams,
    params,
    pc.weight_offsets[1],
    pc.bias_offsets[1],
    CoopVecMatrixLayout::InferencingOptimal,
    CoopVecComponentType::Float16
  );
  hiddenParams = rtxns::relu(hiddenParams);

  outputParams = rtxns::LinearOp<half, 3, 32>(
    hiddenParams,
    params,
    pc.weight_offsets[2],
    pc.bias_offsets[2],
    CoopVecMatrixLayout::InferencingOptimal,
    CoopVecComponentType::Float16
  );
  outputParams = rtxns::sigmoid(outputParams);

  let color = float3(outputParams[0], outputParams[1], outputParams[2]);
  let linearColor = srgbToLinear(color);
  return {
    { linearColor, 1.0 },
  };
}
