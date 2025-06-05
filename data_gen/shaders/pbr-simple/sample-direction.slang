import "../rtxns/PCG32";

// -----------------------------------------------------------------------------
// Random rectangle sampler
//   Generates four direction vectors (v1–v4) that point from the shading point
//   to the four corners of a randomly-oriented, randomly-sized rectangle.
//
//   – 方向分布・立体角・アスペクト比・傾斜角をほぼ一様（または対数一様）にばらけさせる
//   – αmax, γmax, tiltMax などは const 値で調整可
// -----------------------------------------------------------------------------

// ────────────── 調整用定数 ──────────────────────────────────────────
static const float kPi = 3.141592653589793f;
static const float kDegToRad = kPi / 180.0f;

// 最大半角（85° ⇒ 5.9 sr 程度）――「90°ギリギリ」は数値的に不安
static const float kAlphaMaxDeg = 85.0f;
static const float kAlphaMaxRad = kAlphaMaxDeg * kDegToRad;
static const float kSinAlphaMax = sin(kAlphaMaxRad);
static const float kK2 = kSinAlphaMax * kSinAlphaMax; // k² = sin²α_max
static const float kLogAspectMax = log(kAspectMax);

// アスペクト比の最大値 |γ| = max(w/h, h/w)
static const float kAspectMax = 16.0f;

// 面法線と中心方向 w の最大傾斜角（台形化の強さ）
static const float kTiltMaxDeg = 180.0f;
static const float kTiltMaxRad = kTiltMaxDeg * kDegToRad;

// ────────────── 補助関数 ───────────────────────────────────────────
inline float3 anyPerpendicular(float3 v)
{
    // 適当な垂直ベクトルを返す
    return (abs(v.z) < 0.999f) ? float3(-v.y, v.x, 0.0f)
                               : float3(0.0f, -v.z, v.y);
}

inline float3 rotateAroundAxis(float3 v, float3 axis, float angle)
{
    // Rodrigues 公式で回転
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0f - c);
}

inline void buildONB(float3 n, out float3 t, out float3 b)
{
    t = normalize(anyPerpendicular(n));
    b = cross(n, t);
}

// ────────────── メインサンプラ ─────────────────────────────────────
func sampleDirection(
    inout rng: PCG32,
    wi: float3, // 中心方向（from P to rectangle centre）
    out v1: float3,
    out v2: float3,
    out v3: float3,
    out v4: float3)
{
    // 0) 正規化
    float3 w = normalize(wi);

    // ─────────────────────────────────────────────────────────────
    // 1) 面法線 n を w から傾斜させる（投影を台形化）
    //    tilt ∈ [0, tiltMax] を球面一様 (PDF ∝ sinθ)、
    //    ψ ∈ [0,2π) を一様サンプル
    float uTilt = rng.nextFloat();
    float vTilt = rng.nextFloat();
    float cosTiltMax = cos(kTiltMaxRad);
    float tilt = acos(1.0f - (1.0f - cosTiltMax) * uTilt); // 逆CDF
    float psi = 2.0f * kPi * vTilt;

    float3 t0, b0;
    buildONB(w, t0, b0); // w 周りの局所基底
    float3 tiltAxis = normalize(cos(psi) * t0 + sin(psi) * b0);
    float3 n = rotateAroundAxis(-w, tiltAxis, tilt); // 面法線

    // 新しい平面基底
    float3 uDir, vDir;
    buildONB(n, uDir, vDir);

    // ─────────────────────────────────────────────────────────────
    // 2) 立体角とアスペクト比から半角 αx, αy を決定
    //    log-uniform にアスペクト比 γ をサンプリングし、
    //    立体角 Ω を一様サンプリングして半角を求める。

    // 2-1) アスペクト比 γ を log-uniform
    float uG = rng.nextFloat();                            // [0,1)
    float gamma = exp((uG * 2.0f - 1.0f) * kLogAspectMax); // 1/γ_max … γ_max

    // 2-2) γ に応じた最大立体角 Ω_max を計算
    //    与えられたアスペクト比 γ のもとで、
    //    どちらか一方の半角がちょうど αmax になる極限の立体角
    float omegaMax = (gamma >= 1.0f)
                         ? (4.0f * kK2 / gamma)
                         : (4.0f * kK2 * gamma);

    // 2-3) Ω を一様サンプリング
    float uO = rng.nextFloat();  // [0,1)
    float omega = omegaMax * uO; // Ω ∈ [0, Ω_max)

    // 2-4) 半角を求める
    float sinAx = sqrt(omega * gamma * 0.25f);  // √(Ωγ/4)
    float sinAy = sqrt(omega / (4.0f * gamma)); // √(Ω/4γ)
    float ax = asin(sinAx);
    float ay = asin(sinAy);

    // ─────────────────────────────────────────────────────────────
    // 3) 平面内で矩形を回転（回転角 φ 一様）
    float phi = 2.0f * kPi * rng.nextFloat();
    float c = cos(phi);
    float s = sin(phi);
    float3 uRot = c * uDir + s * vDir;
    float3 vRot = -s * uDir + c * vDir;

    // ─────────────────────────────────────────────────────────────
    // 4) 長さスケールを決めて四隅へ射出
    //    距離 dp = 1 とし、w と n の成す角 δ で foreshortening を補正
    float dp = 1.0f;
    float cosDelta = max(0.001f, dot(-w, n)); // 0 に近いと発散するので下限
    // cosDeltaで割るのは長方形が斜めの場合に実長にスケールを保つため。
    float a = dp * tan(ax) / cosDelta; // u 方向半長
    float b = dp * tan(ay) / cosDelta; // v 方向半長

    float3 C = -w * dp; // 矩形中心

    float3 Q1 = C + a * uRot + b * vRot; // 反時計回り (+,+)
    float3 Q2 = C + a * uRot - b * vRot; // (+,−)
    float3 Q3 = C - a * uRot - b * vRot; // (−,−)
    float3 Q4 = C - a * uRot + b * vRot; // (−,+)

    // ─────────────────────────────────────────────────────────────
    // 5) 最大の長さで割って返却
    let maxLength = max(max(length(Q1), length(Q2)),
                        max(length(Q3), length(Q4)));
    v1 = Q1 / maxLength;
    v2 = Q2 / maxLength;
    v3 = Q3 / maxLength;
    v4 = Q4 / maxLength;

    // ─────────────────────────────────────────────────────────────
    // 6) ランダムで逆順に並べ替える
    if (rng.nextFloat() < 0.5f)
    {
        float3 temp = v1;
        v1 = v4;
        v4 = temp;
        temp = v2;
        v2 = v3;
        v3 = temp;
    }
}
