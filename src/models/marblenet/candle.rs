//! Candle implementation of `MarbleNet` VAD.

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_onnx::onnx::{tensor_proto, GraphProto, ModelProto, TensorProto};
use log::debug;

use crate::marblenet_frontend::{speech_probability, MarbleNetFrontend, N_MELS};

pub struct MarbleNet {
    model: candle_onnx::onnx::ModelProto,
    frontend: MarbleNetFrontend,
    device: candle_core::Device,
}

impl MarbleNet {
    /// Loads a `MarbleNet` model for Candle inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX model cannot be read.
    pub fn new(model_path: PathBuf, device: candle_core::Device, _debug: bool) -> Result<Self> {
        let mut model = candle_onnx::read_file(model_path)?;
        expand_dynamic_quantized_ops(&mut model)?;
        Ok(Self {
            model,
            frontend: MarbleNetFrontend::new(),
            device,
        })
    }

    /// Computes frame-level speech probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction or Candle evaluation fails.
    pub fn speech_probabilities(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        let (features, feature_frames) = self.frontend.extract(waveform)?;
        if feature_frames == 0 {
            return Ok(Vec::new());
        }

        let input = Tensor::from_vec(features, (1, N_MELS, feature_frames), &self.device)?;
        debug!(
            "MarbleNet input: {:?}, dtype: {:?}",
            input.shape(),
            input.dtype()
        );
        let outputs = candle_onnx::simple_eval(
            &self.model,
            HashMap::from_iter([("audio_signal".to_string(), input)]),
        )
        .context("failed to evaluate MarbleNet with Candle")?;
        let output = outputs
            .get("outputs")
            .context("MarbleNet model did not return an 'outputs' tensor")?;
        let dimensions = output.dims();
        anyhow::ensure!(
            dimensions.len() == 3 && dimensions[0] == 1 && dimensions[2] == 2,
            "MarbleNet returned logits with shape {dimensions:?}; expected [1, frames, 2]"
        );
        let class_count = dimensions[2];
        let logits = output.flatten_all()?.to_vec1::<f32>()?;
        let probabilities = logits
            .chunks_exact(class_count)
            .map(speech_probability)
            .collect::<Result<Vec<_>>>()?;

        debug!(
            "MarbleNet produced {} frame probabilities",
            probabilities.len()
        );
        Ok(probabilities)
    }
}

fn expand_dynamic_quantized_ops(model: &mut ModelProto) -> Result<()> {
    let graph = model
        .graph
        .as_mut()
        .context("MarbleNet ONNX model does not contain a graph")?;
    let dynamic_nodes: Vec<_> = graph
        .node
        .iter()
        .filter(|node| node.op_type == "DynamicQuantizeLinear")
        .cloned()
        .collect();
    if dynamic_nodes.is_empty() {
        return Ok(());
    }

    let unit_scale = "__marblenet_candle_dynamic_scale".to_string();
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
            "invalid MarbleNet DynamicQuantizeLinear node"
        );
        replacements.insert(node.output[0].clone(), node.input[0].clone());
        // The graph multiplies each integer result by the activation and
        // weight scales. Feeding the original floating-point activation and a
        // unit activation scale preserves that downstream rescaling.
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
        let parameter_axis = match node.op_type.as_str() {
            "ConvInteger" => 0,
            "MatMulInteger" => {
                let weight_name = node
                    .input
                    .get(1)
                    .context("invalid MarbleNet MatMulInteger node")?;
                graph
                    .initializer
                    .iter()
                    .find(|tensor| tensor.name == *weight_name)
                    .context("MarbleNet quantized matrix weight is missing")?
                    .dims
                    .len()
                    .checked_sub(1)
                    .context("MarbleNet quantized matrix weight has no dimensions")?
            }
            _ => continue,
        };
        anyhow::ensure!(
            node.input.len() >= 2,
            "invalid MarbleNet quantized inference node"
        );
        let weight_name = node.input[1].clone();
        let zero_points = node
            .input
            .get(3)
            .filter(|name| !name.is_empty())
            .and_then(|name| integer_initializers.get(name))
            .map_or(&[0_i32][..], Vec::as_slice);
        let weight = graph
            .initializer
            .iter_mut()
            .find(|tensor| tensor.name == weight_name)
            .with_context(|| format!("MarbleNet quantized weight '{weight_name}' is missing"))?;
        // Center the integer weights here; their original scale remains in
        // the graph immediately after the converted Conv/MatMul operation.
        dequantize_weight(weight, zero_points, parameter_axis)?;
        node.op_type = if node.op_type == "ConvInteger" {
            "Conv".to_string()
        } else {
            "MatMul".to_string()
        };
        node.input.truncate(2);
    }

    finish_quantized_expansion(graph)
}

fn finish_quantized_expansion(graph: &mut GraphProto) -> Result<()> {
    anyhow::ensure!(
        !graph.node.iter().any(|node| matches!(
            node.op_type.as_str(),
            "DynamicQuantizeLinear" | "ConvInteger" | "MatMulInteger"
        )),
        "MarbleNet graph contains an unmatched quantized operator"
    );
    let referenced: std::collections::HashSet<&str> = graph
        .node
        .iter()
        .flat_map(|node| node.input.iter().map(String::as_str))
        .chain(graph.output.iter().map(|output| output.name.as_str()))
        .collect();
    graph
        .initializer
        .retain(|tensor| referenced.contains(tensor.name.as_str()));
    Ok(())
}

fn dequantize_weight(weight: &mut TensorProto, zero_points: &[i32], axis: usize) -> Result<()> {
    let values = integer_values(weight)?.context("MarbleNet weight is not INT8 or UINT8")?;
    let axis_size = weight
        .dims
        .get(axis)
        .copied()
        .context("MarbleNet quantized weight has an invalid parameter axis")?;
    let axis_size = usize::try_from(axis_size).context("invalid MarbleNet weight dimension")?;
    anyhow::ensure!(
        zero_points.len() == 1 || zero_points.len() == axis_size,
        "MarbleNet weight zero point has {} values; expected 1 or {axis_size}",
        zero_points.len()
    );
    let inner_size = weight.dims[axis + 1..]
        .iter()
        .try_fold(1_usize, |size, dimension| {
            size.checked_mul(usize::try_from(*dimension).ok()?)
        })
        .context("invalid MarbleNet quantized tensor shape")?;
    weight.float_data = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let parameter_index = if zero_points.len() == 1 {
                0
            } else {
                (index / inner_size) % axis_size
            };
            let centered = i16::try_from(value - zero_points[parameter_index])
                .context("MarbleNet quantized weight exceeds the supported integer range")?;
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
        .context("invalid MarbleNet tensor data type")?;
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
    fn dequantizes_signed_convolution_weights() {
        let mut weight = TensorProto {
            dims: vec![2, 1, 2],
            data_type: tensor_proto::DataType::Int8 as i32,
            raw_data: vec![u8::MAX, 0, 1, 2],
            ..TensorProto::default()
        };

        dequantize_weight(&mut weight, &[-1, 1], 0).unwrap();

        assert_eq!(weight.data_type, tensor_proto::DataType::Float as i32);
        assert_eq!(weight.float_data, [0.0, 1.0, 0.0, 1.0]);
        assert!(weight.raw_data.is_empty());
    }
}
