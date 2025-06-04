import "../rtxns/PCG32";

func intersectZPlane(float3 p1, float3 p2)->float3
{
    let t = -p1.z / max(p2.z - p1.z, 1e-5); // avoid div 0
    return lerp(p1, p2, saturate(t));
}

func clipQuadAgainstZPlane(float3 quad[4], out float3 clipped[5], out int outCount)
{
    uint inCount = 4;

    outCount = 0;
    for (uint i = 0; i < inCount; ++i)
    {
        float3 curr = quad[i];
        float3 prev = quad[(i + inCount - 1) % inCount];

        bool currInside = curr.z >= 0.0;
        bool prevInside = prev.z >= 0.0;

        if (currInside && prevInside)
        {
            clipped[outCount++] = curr;
        }
        else if (currInside && !prevInside)
        {
            clipped[outCount++] = intersectZPlane(prev, curr);
            clipped[outCount++] = curr;
        }
        else if (!currInside && prevInside)
        {
            clipped[outCount++] = intersectZPlane(prev, curr);
        }
    }
}

func triangulatePolygon(
    float3 poly[5],
    int count,
    out Triangle triangles[3],
    out uint triCount)
{
    if (count < 3)
    {
        triCount = 0;
        return;
    }
    triCount = count - 2;
    for (int i = 0; i < triCount; ++i)
    {
        triangles[i].v0 = poly[0];
        triangles[i].v1 = poly[i + 1];
        triangles[i].v2 = poly[i + 2];
    }
}

public struct Triangle
{
    float3 v0;
    float3 v1;
    float3 v2;

    func area()->float
    {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        return length(cross(edge1, edge2)) * 0.5;
    }

    public func randomBarycentric(inout PCG32 rng)->float3
    {
        float u = rng.nextFloat();
        float v = rng.nextFloat();
        if (u + v > 1.0)
        {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        return float3(1.0 - u - v, u, v);
    }

    public func normal()->float3
    {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        return normalize(cross(edge1, edge2));
    }

    public func barycentricToPosition(float3 bary)->float3
    {
        return v0 * bary.x + v1 * bary.y + v2 * bary.z;
    }
};

public struct WeightedTriangles
{
    Triangle triangles[3];
    float weights[3];
    public float totalArea;
    public int triCount;

    public func sample(inout rng: PCG32)->Triangle
    {
        float r = rng.nextFloat();
        float acc = 0.0;
        for (uint i = 0; i < 3; ++i)
        {
            acc += weights[i];
            if (r < acc)
                return triangles[i];
        }
        return triangles[0]; // fallback
    }

    public static func clipAndTriangulateQuad(
        float3 v1,
        float3 v2,
        float3 v3,
        float3 v4)
        ->WeightedTriangles
    {
        float3 quad[4] = { v1, v2, v3, v4 };
        float3 clipped[5];
        int clipCount;
        clipQuadAgainstZPlane(quad, clipped, clipCount);

        Triangle triangles[3];
        int triCount;
        triangulatePolygon(clipped, clipCount, triangles, triCount);

        float total = 0.0;
        float weights[3] = { 0.0, 0.0, 0.0 };
        for (int i = 0; i < triCount; ++i)
        {
            weights[i] = triangles[i].area();
            total += weights[i];
        }

        if (total > 0.0)
        {
            for (int i = 0; i < 3; ++i)
                weights[i] /= total;
        }

        WeightedTriangles result;
        [unroll]
        for (int i = 0; i < 3; ++i)
        {
            result.triangles[i] = triangles[i];
            result.weights[i] = weights[i];
        }
        result.totalArea = total;
        result.triCount = triCount;

        return result;
    }
};
