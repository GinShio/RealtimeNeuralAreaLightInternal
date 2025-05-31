import "../rtxns/CooperativeVectorAutoDiff";
import "../rtxns/CooperativeVectorDerivatives";
import "../rtxns/CooperativeVectorFunctions";

// Function to duplicate CoopVec for use twice.
// This is because using CoopVec twice with different LinearOp, etc.,
// results in incorrect gradient calculations.
public func duplicateCoopVec<let K : int>(
    latentVector: CoopVec<half, K>)
    ->Tuple<CoopVec<half, K>, CoopVec<half, K>>
{
    CoopVec<half, K> first;
    CoopVec<half, K> second;
    [ForceUnroll]
    for (var i = 0; i < K; i++)
    {
        first[i] = latentVector[i];
        second[i] = latentVector[i];
    }
    return makeTuple(first, second);
}

[BackwardDerivativeOf(duplicateCoopVec)]
public func duplicateCoopVec_Backward<let K : int>(
    inout latentVector: DifferentialPair<CoopVec<half, K>>,
    grad: Tuple<CoopVec<half, K>, CoopVec<half, K>>.Differential)
{
    latentVector = diffPair(
        latentVector.p,
        grad._0 + grad._1);
}

[BackwardDerivativeOf(tanh)]
public func tanh_Backward<T : __BuiltinFloatingPointType, let K : int>(
    inout v: DifferentialPair<CoopVec<T, K>>,
    grad: CoopVec<T, K>.Differential)
{
    CoopVec<T, K> gradTanh;
    [ForceUnroll]
    for (var i = 0; i < K; i++)
    {
        gradTanh[i] = grad[i] * (T(1.0) - tanh(v.p[i]) * tanh(v.p[i]));
    }
    v = diffPair(v.p, gradTanh);
}
