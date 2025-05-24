use crate::renderer::{model_data::ModelData, vertex::Vertex, vulkan_state::VulkanState};

// Tangent計算関数
fn compute_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let mut tangents = vec![[0.0; 4]; positions.len()];
    let mut tan1 = vec![[0.0f32; 3]; positions.len()];
    let mut tan2 = vec![[0.0f32; 3]; positions.len()];

    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];

        let uv0 = uvs[i0];
        let uv1 = uvs[i1];
        let uv2 = uvs[i2];

        let x1 = p1[0] - p0[0];
        let y1 = p1[1] - p0[1];
        let z1 = p1[2] - p0[2];
        let x2 = p2[0] - p0[0];
        let y2 = p2[1] - p0[1];
        let z2 = p2[2] - p0[2];

        let s1 = uv1[0] - uv0[0];
        let t1 = uv1[1] - uv0[1];
        let s2 = uv2[0] - uv0[0];
        let t2 = uv2[1] - uv0[1];

        let denom = s1 * t2 - s2 * t1;
        let r = if denom.abs() < 1e-8 { 1.0 } else { 1.0 / denom };

        let s_dir = [
            (t2 * x1 - t1 * x2) * r,
            (t2 * y1 - t1 * y2) * r,
            (t2 * z1 - t1 * z2) * r,
        ];
        let t_dir = [
            (s1 * x2 - s2 * x1) * r,
            (s1 * y2 - s2 * y1) * r,
            (s1 * z2 - s2 * z1) * r,
        ];

        for &i in &[i0, i1, i2] {
            tan1[i][0] += s_dir[0];
            tan1[i][1] += s_dir[1];
            tan1[i][2] += s_dir[2];
            tan2[i][0] += t_dir[0];
            tan2[i][1] += t_dir[1];
            tan2[i][2] += t_dir[2];
        }
    }

    for i in 0..positions.len() {
        let n = normals[i];
        let t = tan1[i];

        // Gram-Schmidt orthogonalize
        let dot_nt = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
        let mut tangent = [
            t[0] - n[0] * dot_nt,
            t[1] - n[1] * dot_nt,
            t[2] - n[2] * dot_nt,
        ];
        let len =
            (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
        if len > 1e-8 {
            tangent[0] /= len;
            tangent[1] /= len;
            tangent[2] /= len;
        }

        // Calculate handedness
        let n_cross_t = [
            n[1] * tangent[2] - n[2] * tangent[1],
            n[2] * tangent[0] - n[0] * tangent[2],
            n[0] * tangent[1] - n[1] * tangent[0],
        ];
        let handedness =
            if (n_cross_t[0] * tan2[i][0] + n_cross_t[1] * tan2[i][1] + n_cross_t[2] * tan2[i][2])
                < 0.0
            {
                -1.0
            } else {
                1.0
            };

        tangents[i] = [tangent[0], tangent[1], tangent[2], handedness];
    }
    tangents
}

pub fn load_sphere(
    state: &mut VulkanState,
    latitude_segments: u32,
    longitude_segments: u32,
) -> ModelData {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Generate vertices
    for lat in 0..=latitude_segments {
        let theta = std::f32::consts::PI * (lat as f32) / (latitude_segments as f32);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=longitude_segments {
            let phi = 2.0 * std::f32::consts::PI * (lon as f32) / (longitude_segments as f32);
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = sin_theta * cos_phi;
            let y = cos_theta;
            let z = sin_theta * sin_phi;

            let u = (lon as f32) / (longitude_segments as f32);
            let v = 1.0 - (lat as f32) / (latitude_segments as f32);

            positions.push([x, y, z]);
            normals.push([x, y, z]);
            uvs.push([u, v]);
        }
    }

    // Generate indices
    let ring = longitude_segments + 1;
    for lat in 0..latitude_segments {
        for lon in 0..longitude_segments {
            let current = lat * ring + lon;
            let next = current + ring;

            indices.push(current);
            indices.push(current + 1);
            indices.push(next);

            indices.push(current + 1);
            indices.push(next + 1);
            indices.push(next);
        }
    }

    // Tangent計算
    let tangents = compute_tangents(&positions, &normals, &uvs, &indices);

    // Vertex構築
    for i in 0..positions.len() {
        vertices.push(Vertex {
            pos: positions[i],
            normal: normals[i],
            tangent: tangents[i],
            uv: uvs[i],
        });
    }

    ModelData::new(state, &vertices, &indices).unwrap()
}
