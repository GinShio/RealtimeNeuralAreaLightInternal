// -----------------------------------------------------------------------------
//  modifyDirection – runtime solid-angle clamp (αmax = 85°)
//  --------------------------------------------------------
//  * 入力:  タンジェント空間で正規化済みの 4 隅方向 v1–v4
//           視線方向 wo（同じタンジェント空間，正規化済み）
//  * 出力:  v1–v4 を in-place 更新（85°≃5.9 sr を超える立体角をクランプ）
//
//  完全対称で v1+v2+v3+v4 ≈ 0 になる場合の代表方向 w は
//  「鏡面反射方向 wr = reflect(−wo, n=(0,0,1))」を用いる。
// -----------------------------------------------------------------------------

static const float kPi = 3.141592653589793f;
static const float kAlphaMaxDeg = 85.0f; // ← 適宜変更
static const float kAlphaMaxRad = kAlphaMaxDeg * kPi / 180.0f;
static const float3 kNormalTS = float3(0.0f, 0.0f, 1.0f); // タンジェント空間法線

// ────────────── ユーティリティ ─────────────────────────────────────
inline float3 safeNormalize(float3 v)
{
    return (length(v) > 1e-6f) ? normalize(v) : kNormalTS;
}

// 線形フォールバック付き SLERP（球面線形補間）
inline float3 slerp(float3 a, float3 b, float t)
{
    float cosT = clamp(dot(a, b), -1.0f, 1.0f);

    // きわめて接近 → LERP で十分
    if (cosT > 0.9995f)
        return safeNormalize(lerp(a, b, t));

    float theta = acos(cosT);
    float sinTheta = sin(theta);
    float w1 = sin((1.0f - t) * theta) / sinTheta;
    float w2 = sin(t * theta) / sinTheta;

    return safeNormalize(a * w1 + b * w2);
}

// HLSL/Slang 標準 reflect(I,N) と同じ定義（念のため明示）
inline float3 reflectDir(float3 I, float3 N)
{
    return I - 2.0f * dot(N, I) * N;
}

// ────────────── メイン関数 ─────────────────────────────────────────
func modifyVertices(
    wo: float3, // 視線（outgoing）方向・正規化済み
    inout v1: float3,
    inout v2: float3,
    inout v3: float3,
    inout v4: float3)
{
    // 0) 代表方向 w を計算
    float3 w = v1 + v2 + v3 + v4;
    if (length(w) < 1e-4f) // 完全対称 → 鏡面反射方向を採用
    {
        float3 wr = reflectDir(-wo, kNormalTS); // 入射方向への鏡面反射
        w = safeNormalize(wr);
    }
    else
    {
        w = safeNormalize(w);
    }

    // 1) 各隅との角度 θ_i を取得
    float angle1 = acos(clamp(dot(w, v1), -1.0f, 1.0f));
    float angle2 = acos(clamp(dot(w, v2), -1.0f, 1.0f));
    float angle3 = acos(clamp(dot(w, v3), -1.0f, 1.0f));
    float angle4 = acos(clamp(dot(w, v4), -1.0f, 1.0f));

    float thetaMax = max(max(angle1, angle2), max(angle3, angle4));

    // 2) θmax > αmax なら全隅を w 側へ縮める
    if (thetaMax > kAlphaMaxRad)
    {
        float ratio = kAlphaMaxRad / thetaMax; // 0‥1
        v1 = slerp(w, v1, ratio);
        v2 = slerp(w, v2, ratio);
        v3 = slerp(w, v3, ratio);
        v4 = slerp(w, v4, ratio);
    }

    // v1–v4 は in-place で更新済み
}
