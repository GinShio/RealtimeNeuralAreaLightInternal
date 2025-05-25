use anyhow::Result;
use ash::{nv::cooperative_vector, vk};
use half::f16;
use rand_distr::{Distribution, StandardNormal};

const COOPERATIVE_VECTOR_MATRIX_ALIGNMENT: u32 = 64;
const COOPERATIVE_VECTOR_VECTOR_ALIGNMENT: u32 = 16;

pub struct Network {
    pub data: Vec<u8>,
    pub weight_offsets: Vec<u32>,
    pub bias_offsets: Vec<u32>,
}
impl Network {
    fn align_to(alignment: u32, offset: u32) -> u32 {
        offset.div_ceil(alignment) * alignment
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
            .dst_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::TrainingOptimal)
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

    fn convert_matrix_data_to_training_optimal(
        cooperative_vector_fn: &cooperative_vector::Device,
        src_data: &[f32],
        rows: u32,
        cols: u32,
        required_size: usize,
    ) -> Result<Vec<u8>> {
        let mut result = vec![0u8; required_size];

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
            .dst_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::TrainingOptimal)
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

    pub fn from_dimensions(
        cooperative_vector_fn: &cooperative_vector::Device,
        dimensions: &[u32],
    ) -> Result<Self> {
        let dimensions = dimensions.iter().copied().collect::<Vec<_>>();

        let mut offset = 0;
        let mut weight_sizes = vec![];
        let mut weight_offsets = vec![];
        let mut bias_sizes = vec![];
        let mut bias_offsets = vec![];
        for item in dimensions.windows(2) {
            let in_features = item[0];
            let out_features = item[1];

            let weight_size =
                Self::query_matrix_byte_size(cooperative_vector_fn, out_features, in_features)?;
            let bias_size = out_features * std::mem::size_of::<f16>() as u32;

            weight_sizes.push(weight_size);
            bias_sizes.push(bias_size);

            offset = Self::align_to(COOPERATIVE_VECTOR_MATRIX_ALIGNMENT, offset);
            weight_offsets.push(offset);
            offset += weight_size;

            offset = Self::align_to(COOPERATIVE_VECTOR_VECTOR_ALIGNMENT, offset);
            bias_offsets.push(offset);
            offset += bias_size;
        }

        let mut rng = rand::rng();

        let mut data = vec![0u8; offset as usize];
        for (i, item) in dimensions.windows(2).enumerate() {
            let in_feature = item[0];
            let out_feature = item[1];

            let std = (2.0 / in_feature as f32).sqrt();
            let weights: Vec<f32> = (0..(in_feature as usize * out_feature as usize))
                .map(|_| {
                    let value: f32 = StandardNormal.sample(&mut rng);
                    value * std
                })
                .collect();
            let biases = vec![0.0; out_feature as usize];

            let weights_bytes = Self::convert_matrix_data_to_training_optimal(
                cooperative_vector_fn,
                &weights,
                out_feature,
                in_feature,
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

pub struct TrainedNetwork {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
    pub in_features: Vec<u32>,
    pub out_features: Vec<u32>,
}
impl TrainedNetwork {
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
            .src_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::TrainingOptimal)
            .src_stride(stride)
            .src_size(0)
            .src_data(vk::DeviceOrHostAddressConstKHR {
                host_address: std::ptr::null(),
            })
            .dst_component_type(vk::ComponentTypeNV::FLOAT16)
            .dst_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::RowMajor)
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

    fn convert_matrix_data_from_training_optimal(
        cooperative_vector_fn: &cooperative_vector::Device,
        src_data: &[u8],
        rows: u32,
        cols: u32,
        required_size: usize,
    ) -> Result<Vec<f32>> {
        let mut data = vec![0u8; required_size];

        let size = (rows * cols) as usize * std::mem::size_of::<f16>();

        let src_stride = rows as u64 * std::mem::size_of::<f16>() as u64;
        let dst_stride = cols as u64 * std::mem::size_of::<f16>() as u64;

        let mut actual_size = size;

        let info = cooperative_vector::ConvertCooperativeVectorMatrixInfoNV::default()
            .num_rows(rows)
            .num_columns(cols)
            .src_component_type(vk::ComponentTypeNV::FLOAT16)
            .src_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::TrainingOptimal)
            .src_stride(src_stride)
            .src_size(required_size)
            .src_data(vk::DeviceOrHostAddressConstKHR {
                host_address: src_data.as_ptr() as *const _,
            })
            .dst_component_type(vk::ComponentTypeNV::FLOAT16)
            .dst_layout(cooperative_vector::CooperativeVectorMatrixLayoutNV::RowMajor)
            .dst_stride(dst_stride)
            .dst_size(&mut actual_size)
            .dst_data(vk::DeviceOrHostAddressKHR {
                host_address: data.as_mut_ptr() as *mut _,
            });
        unsafe {
            cooperative_vector_fn.convert_cooperative_vector_matrix_nv(&info)?;
        }

        let dst_size = actual_size as usize / std::mem::size_of::<f16>();
        let mut result = Vec::<f32>::with_capacity(dst_size);
        for bytes in data.chunks(2) {
            let bytes = [bytes[0], bytes[1]];
            let f16_value = half::f16::from_ne_bytes(bytes);
            let f32_value = f16_value.to_f32();
            result.push(f32_value);
        }

        Ok(result)
    }

    pub fn from_data(
        cooperative_vector_fn: &cooperative_vector::Device,
        data: &[u8],
        weight_offsets: &[u32],
        bias_offsets: &[u32],
        dimensions: &[u32],
    ) -> Result<Self> {
        let mut weights = vec![];
        let mut biases = vec![];
        let mut in_features = vec![];
        let mut out_features = vec![];

        for (i, item) in dimensions.windows(2).enumerate() {
            let in_feature = item[0];
            let out_feature = item[1];

            let weight_size =
                Self::query_matrix_byte_size(cooperative_vector_fn, out_feature, in_feature)?;
            let bias_size = out_feature * std::mem::size_of::<f16>() as u32;

            let weight_data =
                &data[weight_offsets[i] as usize..(weight_offsets[i] + weight_size) as usize];
            let bias_data = &data[bias_offsets[i] as usize..(bias_offsets[i] + bias_size) as usize];

            let converted_weights = Self::convert_matrix_data_from_training_optimal(
                cooperative_vector_fn,
                weight_data,
                out_feature,
                in_feature,
                weight_size as usize,
            )?;
            let converted_biases_f16 = bytemuck::cast_slice::<u8, f16>(bias_data).to_vec();
            let converted_biases = converted_biases_f16
                .iter()
                .map(|&b| b.to_f32())
                .collect::<Vec<f32>>();

            weights.push(converted_weights);
            biases.push(converted_biases);
            in_features.push(in_feature);
            out_features.push(out_feature);
        }

        Ok(Self {
            weights,
            biases,
            in_features,
            out_features,
        })
    }

    pub fn save_network(&self, path: &str) -> Result<()> {
        use serde_json::json;
        use std::fs::File;
        use std::io::Write;

        let path = std::path::Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut layers = Vec::new();
        for ((in_features, out_features), (weights, biases)) in self
            .in_features
            .iter()
            .zip(self.out_features.iter())
            .zip(self.weights.iter().zip(self.biases.iter()))
        {
            layers.push(json!({
                "in_features": in_features,
                "out_features": out_features,
                "weights": weights,
                "bias": biases,
                "type": "linear",
            }));
        }
        let model = json!({
            "input": self.in_features.first().copied().unwrap_or(0),
            "output": self.out_features.last().copied().unwrap_or(0),
            "layers": layers,
        });
        let root = json!({ "model": model });

        let json_str = serde_json::to_string_pretty(&root)?;
        let mut file = File::create(path)?;
        file.write_all(json_str.as_bytes())?;
        Ok(())
    }
}
