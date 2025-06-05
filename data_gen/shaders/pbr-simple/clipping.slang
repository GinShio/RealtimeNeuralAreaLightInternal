import "../rtxns/PCG32";

static const float CLIP_Z = 0.0;

func intersectZPlane(float3 a, float3 b)->float3
{
    float t = (CLIP_Z - a.z) / (b.z - a.z);
    return lerp(a, b, t);
}

func clipQuadAgainstZPlane(float3 quad[4], out float3 clipped[6], out int outCount)
{
    outCount = 0;
    for (int i = 0; i < 4; ++i)
    {
        float3 curr = quad[i];
        float3 prev = quad[(i + 3) % 4];

        bool currInside = curr.z >= CLIP_Z;
        bool prevInside = prev.z >= CLIP_Z;

        if (currInside != prevInside) // 符号が異なる＝交差
            clipped[outCount++] = intersectZPlane(prev, curr);

        if (currInside) // 現頂点が内側なら push
            clipped[outCount++] = curr;
    }
}

func triangulatePolygon(
    float3 poly[6],
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
    triCount = 0;
    for (int i = 0; i < count; ++i)
    {
        var triangle = Triangle();
        triangle.v0 = poly[0];
        triangle.v1 = poly[i];
        triangle.v2 = poly[i + 1];
        if (triangle.area() < 1e-12)
        {
            continue;
        }
        triangles[triCount++] = triangle;
    }
}

public struct Triangle
{
    public float3 v0;
    public float3 v1;
    public float3 v2;

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
        const float EPS_DET = 1e-9;
        const float EPS_BARY = 1e-5;

        hitPos = float3(0.0, 0.0, 0.0);
        hitNormal = float3(0.0, 0.0, 0.0);

        let epsilon = 1e-9;

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;

        let pvec = cross(v, edge2);
        let det = dot(edge1, pvec);

        if (abs(det) < EPS_DET)
        {
            if (abs(dot(v0, normal())) < 1e-5)
            {
                hitNormal = normal();
                return true; // レイが三角形の平面上にある
            }

            return false; // レイと面が平行
        }

        let invDet = 1.0 / det;
        let tvec = -v0; // レイの始点は原点
        let u = dot(tvec, pvec) * invDet;
        if (u < 0.0 - EPS_BARY || u > 1.0 + EPS_BARY)
        {
            return false;
        }

        let qvec = cross(tvec, edge1);
        let vval = dot(v, qvec) * invDet;
        if (vval < 0.0 - EPS_BARY || u + vval > 1.0 + EPS_BARY)
        {
            return false; // 三角形の外側
        }

        let t = dot(edge2, qvec) * invDet;
        if (t < epsilon)
        {
            return false; // 三角形の後ろにある
        }

        hitPos = v * t;
        hitNormal = normal();
        return true;
    }
};

public struct WeightedTriangles
{
    public Triangle triangles[3];
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
        float3 clipped[6];
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
