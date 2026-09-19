//! Candle implementation of FSMN-VAD.

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::{tensor_proto, ModelProto, TensorProto};
use log::debug;

use crate::fsmn_vad_frontend::{FsmnVadFrontend, FEATURE_DIM};

const CACHE_COUNT: usize = 4;
const CACHE_DIM: usize = 128;
const CACHE_ORDER: usize = 19;
const MAX_CHUNK_FRAMES: usize = 6_000;

pub struct FsmnVad {
    model: candle_onnx::onnx::ModelProto,
    frontend: FsmnVadFrontend,
    device: Device,
}

impl FsmnVad {
    /// Loads an FSMN model and its preprocessing sidecar for Candle inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or CMVN sidecar cannot be loaded.
    pub fn new(model_path: PathBuf, device: Device, _debug: bool) -> Result<Self> {
        let frontend = FsmnVadFrontend::from_model_path(&model_path)?;
        let mut model = candle_onnx::read_file(model_path)?;
        expand_dynamic_quantized_matmul(&mut model)?;
        Ok(Self {
            model,
            frontend,
            device,
        })
    }

    /// Computes a speech probability for every FSMN frame.
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction, inference, or tensor validation fails.
    pub fn speech_probabilities(&mut self, waveform: &[f32]) -> Result<Vec<f32>> {
        let features = self.frontend.extract(waveform)?;
        if features.is_empty() {
            return Ok(Vec::new());
        }

        let cache_shape = (1, CACHE_DIM, CACHE_ORDER, 1);
        let empty_cache = Tensor::zeros(cache_shape, candle_core::DType::F32, &self.device)?;
        let mut caches: [Tensor; CACHE_COUNT] = std::array::from_fn(|_| empty_cache.clone());
        let mut probabilities = Vec::with_capacity(features.len() / FEATURE_DIM);

        for chunk in features.chunks(MAX_CHUNK_FRAMES * FEATURE_DIM) {
            let frame_count = chunk.len() / FEATURE_DIM;
            let speech = Tensor::from_slice(chunk, (1, frame_count, FEATURE_DIM), &self.device)?;
            debug!(
                "FSMN-VAD input: {:?}, dtype: {:?}",
                speech.shape(),
                speech.dtype()
            );
            let mut inputs = HashMap::from_iter([("speech".to_string(), speech)]);
            for (index, cache) in caches.iter().enumerate() {
                inputs.insert(format!("in_cache{index}"), cache.clone());
            }

            let outputs = candle_onnx::simple_eval(&self.model, inputs)
                .context("failed to evaluate FSMN-VAD with Candle")?;
            let logits = outputs
                .get("logits")
                .context("FSMN-VAD model did not return a 'logits' tensor")?;
            let dimensions = logits.dims();
            anyhow::ensure!(
                dimensions.len() == 3 && dimensions[0] == 1 && dimensions[1] == frame_count,
                "FSMN-VAD returned logits with shape {dimensions:?}; expected [1, {frame_count}, classes]"
            );
            let class_count = dimensions[2];
            anyhow::ensure!(class_count >= 2, "FSMN-VAD returned fewer than two classes");
            let scores = logits.flatten_all()?.to_vec1::<f32>()?;
            anyhow::ensure!(
                scores.len() == frame_count * class_count,
                "FSMN-VAD returned an inconsistent logits tensor"
            );
            probabilities.extend(
                scores
                    .chunks_exact(class_count)
                    .map(|frame_scores| (1.0 - frame_scores[0]).clamp(0.0, 1.0)),
            );

            for (index, cache) in caches.iter_mut().enumerate() {
                let name = format!("out_cache{index}");
                let next_cache = outputs
                    .get(&name)
                    .with_context(|| format!("FSMN-VAD model did not return '{name}'"))?;
                anyhow::ensure!(
                    next_cache.dims() == [1, CACHE_DIM, CACHE_ORDER, 1],
                    "FSMN-VAD returned {name} with shape {:?}; expected [1, {CACHE_DIM}, {CACHE_ORDER}, 1]",
                    next_cache.dims()
                );
                cache.clone_from(next_cache);
            }
        }

        debug!(
            "FSMN-VAD produced {} frame probabilities",
            probabilities.len()
        );
        Ok(probabilities)
    }
}

fn expand_dynamic_quantized_matmul(model: &mut ModelProto) -> Result<()> {
    let graph = model
        .graph
        .as_mut()
        .context("FSMN-VAD ONNX model does not contain a graph")?;
    let dynamic_nodes: Vec<_> = graph
        .node
        .iter()
        .filter(|node| node.op_type == "DynamicQuantizeLinear")
        .cloned()
        .collect();
    if dynamic_nodes.is_empty() {
        return Ok(());
    }

    let unit_scale = "__fsmn_candle_dynamic_scale".to_string();
    graph.initializer.push(TensorProto {
        data_type: tensor_proto::DataType::Float as i32,
        float_data: vec![1.0],
        name: unit_scale.clone(),
        ..TensorProto::default()
    });

    let mut replacements = HashMap::new();
    for node in &dynamic_nodes {
        anyhow::ensure!(
            node.input.len() == 1 && node.output.len() == 3,
            "invalid FSMN-VAD DynamicQuantizeLinear node"
        );
        replacements.insert(node.output[0].clone(), node.input[0].clone());
        replacements.insert(node.output[1].clone(), unit_scale.clone());
    }
    graph
        .node
        .retain(|node| node.op_type != "DynamicQuantizeLinear");

    let integer_initializers: HashMap<String, Vec<i32>> = graph
        .initializer
        .iter()
        .filter_map(|tensor| {
            integer_values(tensor)
                .transpose()
                .map(|values| values.map(|values| (tensor.name.clone(), values)))
        })
        .collect::<Result<_>>()?;

    for node in &mut graph.node {
        for input in &mut node.input {
            if let Some(replacement) = replacements.get(input) {
                input.clone_from(replacement);
            }
        }
        if node.op_type != "MatMulInteger" {
            continue;
        }
        anyhow::ensure!(node.input.len() >= 2, "invalid FSMN-VAD MatMulInteger node");
        let weight_name = &node.input[1];
        let zero_points = node
            .input
            .get(3)
            .filter(|name| !name.is_empty())
            .and_then(|name| integer_initializers.get(name))
            .map_or(&[0_i32][..], Vec::as_slice);
        let weight = graph
            .initializer
            .iter_mut()
            .find(|tensor| tensor.name == *weight_name)
            .with_context(|| format!("FSMN-VAD quantized weight '{weight_name}' is missing"))?;
        dequantize_weight(weight, zero_points)?;
        node.op_type = "MatMul".to_string();
        node.input.truncate(2);
    }
    Ok(())
}

fn dequantize_weight(weight: &mut TensorProto, zero_points: &[i32]) -> Result<()> {
    let values = integer_values(weight)?.context("FSMN-VAD weight is not INT8 or UINT8")?;
    let output_channels = weight
        .dims
        .last()
        .copied()
        .context("FSMN-VAD quantized weight has no dimensions")?;
    let output_channels =
        usize::try_from(output_channels).context("invalid FSMN-VAD weight dimension")?;
    anyhow::ensure!(
        zero_points.len() == 1 || zero_points.len() == output_channels,
        "FSMN-VAD weight zero point has {} values; expected 1 or {output_channels}",
        zero_points.len()
    );
    weight.float_data = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let zero_point = zero_points[index % zero_points.len()];
            let centered = i16::try_from(value - zero_point)
                .context("FSMN-VAD quantized weight exceeds the supported INT8 range")?;
            Ok(f32::from(centered))
        })
        .collect::<Result<Vec<_>>>()?;
    weight.data_type = tensor_proto::DataType::Float as i32;
    weight.raw_data.clear();
    weight.int32_data.clear();
    Ok(())
}

fn integer_values(tensor: &TensorProto) -> Result<Option<Vec<i32>>> {
    let data_type = tensor_proto::DataType::try_from(tensor.data_type)
        .context("invalid FSMN-VAD tensor data type")?;
    let values = match data_type {
        tensor_proto::DataType::Int8 => {
            if tensor.raw_data.is_empty() {
                tensor.int32_data.clone()
            } else {
                tensor
                    .raw_data
                    .iter()
                    .map(|value| i32::from(i8::from_ne_bytes([*value])))
                    .collect()
            }
        }
        tensor_proto::DataType::Uint8 => {
            if tensor.raw_data.is_empty() {
                tensor.int32_data.clone()
            } else {
                tensor
                    .raw_data
                    .iter()
                    .map(|value| i32::from(*value))
                    .collect()
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantizes_signed_weights_with_per_channel_zero_points() {
        let mut weight = TensorProto {
            dims: vec![2, 2],
            data_type: tensor_proto::DataType::Int8 as i32,
            raw_data: vec![u8::MAX, 0, 1, 2],
            ..TensorProto::default()
        };

        dequantize_weight(&mut weight, &[-1, 1]).unwrap();

        assert_eq!(weight.data_type, tensor_proto::DataType::Float as i32);
        assert_eq!(weight.float_data, [0.0, -1.0, 2.0, 1.0]);
        assert!(weight.raw_data.is_empty());
    }
}
