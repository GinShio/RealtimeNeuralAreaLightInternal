use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Result, anyhow};
use ash::{nv::cooperative_vector, vk};
use half::f16;

const COOPERATIVE_VECTOR_MATRIX_ALIGNMENT: usize = 64;
const COOPERATIVE_VECTOR_VECTOR_ALIGNMENT: usize = 16;

pub struct Network {
    pub data: Vec<u8>,
    pub weight_offsets: Vec<u32>,
    pub bias_offsets: Vec<u32>,
}
impl Network {
    fn align_to(alignment: usize, offset: u32) -> u32 {
        let align = alignment as u32;
        ((offset + align - 1) / align) * align
    }

    fn query_matrix_byte_size(
        cooperative_vector_fn: &cooperative_vector::Device,
        rows: u32,
        cols: u32,
    ) -> Result<u32> {
        let mut required_size = 0;
        let stride = cols as u64 * std::mem::size_of::<f16>() as u64;

        let info = cooperative_vector::ConvertCooperativeVectorMatrixInfoNV::default()
            .num_rows(rows)
            .num_columns(cols)
            .src_component_type(vk::ComponentTypeNV::FLOAT16)
            .src_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::RowMajor)
            .src_stride(stride)
            .src_size(0)
            .src_data(vk::DeviceOrHostAddressConstKHR {
                host_address: std::ptr::null(),
            })
            .dst_component_type(vk::ComponentTypeNV::FLOAT16)
            .dst_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::InferencingOptimal)
            .dst_stride(stride)
            .dst_size(&mut required_size)
            .dst_data(vk::DeviceOrHostAddressKHR {
                host_address: std::ptr::null_mut(),
            });
        unsafe {
            cooperative_vector_fn.convert_cooperative_vector_matrix_nv(&info)?;
        }
        Ok(required_size as u32)
    }

    fn convert_matrix_data_to_inferencing_optimal(
        cooperative_vector_fn: &cooperative_vector::Device,
        src_data: &[f32],
        rows: u32,
        cols: u32,
        required_size: usize,
    ) -> Result<Vec<u8>> {
        let mut result = vec![0u8; required_size as usize];

        let size = (rows * cols) as usize * std::mem::size_of::<f16>();

        let mut src_data_f16 = vec![f16::ZERO; src_data.len()];
        for (i, &value) in src_data.iter().enumerate() {
            src_data_f16[i] = f16::from_f32(value);
        }

        let src_stride = cols as u64 * std::mem::size_of::<f16>() as u64;
        let dst_stride = rows as u64 * std::mem::size_of::<f16>() as u64;

        let mut actual_size = required_size;

        let info = cooperative_vector::ConvertCooperativeVectorMatrixInfoNV::default()
            .num_rows(rows)
            .num_columns(cols)
            .src_component_type(vk::ComponentTypeNV::FLOAT16)
            .src_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::RowMajor)
            .src_stride(src_stride)
            .src_size(size)
            .src_data(vk::DeviceOrHostAddressConstKHR {
                host_address: src_data_f16.as_ptr() as *const _,
            })
            .dst_component_type(vk::ComponentTypeNV::FLOAT16)
            .dst_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::InferencingOptimal)
            .dst_stride(dst_stride)
            .dst_size(&mut actual_size)
            .dst_data(vk::DeviceOrHostAddressKHR {
                host_address: result.as_mut_ptr() as *mut _,
            });
        unsafe {
            cooperative_vector_fn.convert_cooperative_vector_matrix_nv(&info)?;
        }

        Ok(result)
    }

    pub fn from_json(
        cooperative_vector_fn: &cooperative_vector::Device,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let json: serde_json::Value = serde_json::from_reader(reader)?;

        let layers = json
            .get("model")
            .and_then(|m| m.get("layers"))
            .and_then(|l| l.as_array())
            .ok_or_else(|| anyhow!("Invalid model format: missing layers"))?;

        let mut json_data = Vec::new();

        for layer in layers {
            if let Some(layer_type) = layer.get("type").and_then(|t| t.as_str()) {
                if layer_type == "linear" {
                    let in_features = layer
                        .get("in_features")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow!("Missing in_features"))?
                        as u32;
                    let out_features = layer
                        .get("out_features")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow!("Missing out_features"))?
                        as u32;
                    let weights = layer
                        .get("weights")
                        .and_then(|w| w.as_array())
                        .ok_or_else(|| anyhow!("Missing weights"))?
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect::<Vec<f32>>();
                    let biases = layer
                        .get("bias")
                        .and_then(|b| b.as_array())
                        .ok_or_else(|| anyhow!("Missing bias"))?
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect::<Vec<f32>>();
                    json_data.push((in_features, out_features, weights, biases));
                }
            }
        }

        let mut offset = 0;
        let mut weight_sizes = vec![];
        let mut weight_offsets = vec![];
        let mut bias_sizes = vec![];
        let mut bias_offsets = vec![];
        for (in_features, out_features, _, _) in &json_data {
            let weight_size =
                Self::query_matrix_byte_size(cooperative_vector_fn, *out_features, *in_features)?;
            let bias_size = *out_features * std::mem::size_of::<f16>() as u32;

            weight_sizes.push(weight_size);
            bias_sizes.push(bias_size);

            offset = Self::align_to(COOPERATIVE_VECTOR_MATRIX_ALIGNMENT, offset);
            weight_offsets.push(offset);
            offset += weight_size;

            offset = Self::align_to(COOPERATIVE_VECTOR_VECTOR_ALIGNMENT, offset);
            bias_offsets.push(offset);
            offset += bias_size;
        }

        let mut data = vec![0u8; offset as usize];
        for (i, (in_features, out_features, weights, biases)) in json_data.iter().enumerate() {
            let weights_bytes = Self::convert_matrix_data_to_inferencing_optimal(
                cooperative_vector_fn,
                weights,
                *out_features,
                *in_features,
                weight_sizes[i] as usize,
            )?;
            let w_off = weight_offsets[i] as usize;
            let w_size = weight_sizes[i] as usize;
            data[w_off..w_off + w_size].copy_from_slice(&weights_bytes[..w_size]);

            let mut bias_f16 = vec![f16::ZERO; biases.len()];
            for (j, &b) in biases.iter().enumerate() {
                bias_f16[j] = f16::from_f32(b);
            }
            let bias_bytes = bytemuck::cast_slice(&bias_f16);

            let b_off = bias_offsets[i] as usize;
            let b_size = bias_sizes[i] as usize;
            data[b_off..b_off + b_size].copy_from_slice(&bias_bytes[..b_size]);
        }

        Ok(Self {
            data,
            weight_offsets,
            bias_offsets,
        })
    }
}
