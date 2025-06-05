import "../rtxns/PCG32";

func intersectZPlane(float3 p1, float3 p2)->float3
{
    // clip with the z = 0.01 plane
    if (p1.z > 0.0 && p2.z <= 0.0)
    {
        let t = (0.01 - p1.z) / max(p1.z - p2.z, 1e-8);
        return lerp(p1, p2, saturate(t));
    }
    else
    {
        let t = (0.01 - p2.z) / max(p2.z - p1.z, 1e-8);
        return lerp(p2, p1, saturate(t));
    }
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
        triangles = { Triangle(), Triangle(), Triangle() };
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

    public func hit(v: float3, out hitPos: float3, out hitNormal: float3)->bool
    {
        hitPos = float3(0.0, 0.0, 0.0);
        hitNormal = float3(0.0, 0.0, 0.0);

        let vv2 = dot(v, cross(v1, v0));
        let vv1 = dot(v, cross(v0, v2));
        let vv0 = dot(v, cross(v2, v1));

        if (vv0 < 0.0 || vv1 < 0.0 || vv2 < 0.0)
        {
            // 三角形の外側にある
            return false;
        }

        let vv = vv0 + vv1 + vv2;
        if (vv < 1e-5)
        {
            // 面に平行
            return false;
        }

        hitPos = barycentricToPosition(float3(vv0, vv1, vv2) / vv);
        hitNormal = normal();
        return true;
    }
};

public struct WeightedTriangles
{
    Triangle triangles[3];
    float weights[3];
    public float totalArea;
    public int triCount;

    public func sampleTriangle(inout rng: PCG32)->Triangle
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

    public func hit(v: float3, out float3 hitPos, out hitNormal: float3)->bool
    {
        hitPos = float3(0.0, 0.0, 0.0);
        hitNormal = float3(0.0, 0.0, 0.0);

        if (totalArea <= 0.0)
        {
            return false;
        }
        let dir = normalize(v);
        for (int i = 0; i < triCount; ++i)
        {
            if (triangles[i].hit(dir, hitPos, hitNormal))
            {
                return true;
            }
        }

        return false;
    }
};
