//! Candle implementation of `PulseVAD`.

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_onnx::onnx::{tensor_proto, ModelProto, TensorProto};
use log::debug;

use crate::{
    pulsevad_frontend::{PulseVadFrontend, N_FRAMES, N_MELS},
    vad_iter::VadModel,
};

pub struct PulseVad {
    model: candle_onnx::onnx::ModelProto,
    frontend: PulseVadFrontend,
    device: candle_core::Device,
}

impl PulseVad {
    /// Loads a `PulseVAD` model for Candle inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX model cannot be read.
    pub fn new(
        model_path: PathBuf,
        device: candle_core::Device,
        _debug: bool,
    ) -> anyhow::Result<Self> {
        let mut model = candle_onnx::read_file(model_path)?;
        expand_qdq(&mut model)?;
        Ok(Self {
            model,
            frontend: PulseVadFrontend::new(),
            device,
        })
    }
}

impl VadModel for PulseVad {
    fn reset(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn probability(&mut self, audio_frame: &[f32]) -> anyhow::Result<f32> {
        let features = self.frontend.extract(audio_frame)?;
        let input = Tensor::from_vec(features, (1, N_MELS, N_FRAMES), &self.device)?;
        let outputs = candle_onnx::simple_eval(
            &self.model,
            HashMap::from_iter([("log_mel".to_string(), input)]),
        )
        .context("failed to evaluate PulseVAD with Candle")?;
        let logits = outputs
            .get("logits")
            .context("PulseVAD model did not return a 'logits' tensor")?;
        anyhow::ensure!(
            matches!(logits.dims(), [2] | [1, 2]),
            "PulseVAD returned logits with shape {:?}; expected [2] or [1, 2]",
            logits.dims()
        );
        let logits = logits.flatten_all()?.to_vec1::<f32>()?;

        let probability = sigmoid(logits[1] - logits[0]);
        debug!("PulseVAD speech probability: {probability:.6}");
        Ok(probability)
    }
}

fn expand_qdq(model: &mut ModelProto) -> Result<()> {
    let graph = model
        .graph
        .as_mut()
        .context("PulseVAD ONNX model does not contain a graph")?;
    if !graph
        .node
        .iter()
        .any(|node| node.op_type == "QuantizeLinear")
    {
        return Ok(());
    }

    let initializers: HashMap<String, TensorProto> = graph
        .initializer
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.clone()))
        .collect();
    let quantize_sources: HashMap<String, String> = graph
        .node
        .iter()
        .filter(|node| node.op_type == "QuantizeLinear")
        .map(|node| {
            anyhow::ensure!(
                node.input.len() >= 2 && node.output.len() == 1,
                "invalid PulseVAD QuantizeLinear node"
            );
            Ok((node.output[0].clone(), node.input[0].clone()))
        })
        .collect::<Result<_>>()?;

    let mut removed_quantize_outputs = std::collections::HashSet::new();
    let mut dequantized_initializers = Vec::new();
    let mut nodes = Vec::with_capacity(graph.node.len());
    for mut node in std::mem::take(&mut graph.node) {
        if node.op_type != "DequantizeLinear" {
            nodes.push(node);
            continue;
        }
        anyhow::ensure!(
            node.input.len() >= 2 && node.output.len() == 1,
            "invalid PulseVAD DequantizeLinear node"
        );

        if let Some(quantized) = initializers.get(&node.input[0]) {
            let scale = initializers
                .get(&node.input[1])
                .with_context(|| format!("PulseVAD scale '{}' is missing", node.input[1]))?;
            let zero_point = node
                .input
                .get(2)
                .filter(|name| !name.is_empty())
                .map(|name| {
                    initializers
                        .get(name)
                        .with_context(|| format!("PulseVAD zero point '{name}' is missing"))
                })
                .transpose()?;
            let axis = node
                .attribute
                .iter()
                .find(|attribute| attribute.name == "axis")
                .map_or(1, |attribute| attribute.i);
            dequantized_initializers.push(dequantize_initializer(
                quantized,
                scale,
                zero_point,
                axis,
                &node.output[0],
            )?);
            continue;
        }

        let source = quantize_sources.get(&node.input[0]).with_context(|| {
            format!("PulseVAD quantized value '{}' has no source", node.input[0])
        })?;
        removed_quantize_outputs.insert(node.input[0].clone());
        node.op_type = "Identity".to_string();
        node.input = vec![source.clone()];
        node.attribute.clear();
        nodes.push(node);
    }
    nodes.retain(|node| {
        node.op_type != "QuantizeLinear"
            || !node
                .output
                .first()
                .is_some_and(|output| removed_quantize_outputs.contains(output))
    });
    anyhow::ensure!(
        !nodes.iter().any(|node| node.op_type == "QuantizeLinear"),
        "PulseVAD graph contains an unmatched QuantizeLinear node"
    );
    graph.node = nodes;
    graph.initializer.extend(dequantized_initializers);
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

fn dequantize_initializer(
    quantized: &TensorProto,
    scale: &TensorProto,
    zero_point: Option<&TensorProto>,
    axis: i64,
    output_name: &str,
) -> Result<TensorProto> {
    let values = integer_values(quantized)?.context("PulseVAD QDQ tensor is not integer")?;
    let scales = float_values(scale)?;
    anyhow::ensure!(!scales.is_empty(), "PulseVAD QDQ scale is empty");
    let zero_points = zero_point
        .map(integer_values)
        .transpose()?
        .flatten()
        .unwrap_or_else(|| vec![0]);
    anyhow::ensure!(
        zero_points.len() == 1 || zero_points.len() == scales.len(),
        "PulseVAD QDQ zero point has {} values; expected 1 or {}",
        zero_points.len(),
        scales.len()
    );

    let rank = quantized.dims.len();
    let axis = if axis < 0 {
        usize::try_from(i64::try_from(rank)? + axis).context("invalid PulseVAD QDQ axis")?
    } else {
        usize::try_from(axis).context("invalid PulseVAD QDQ axis")?
    };
    anyhow::ensure!(axis < rank, "PulseVAD QDQ axis exceeds tensor rank");
    let axis_size = usize::try_from(quantized.dims[axis])
        .context("invalid PulseVAD quantized tensor dimension")?;
    anyhow::ensure!(
        scales.len() == 1 || scales.len() == axis_size,
        "PulseVAD QDQ parameters have {} values; expected 1 or {axis_size}",
        scales.len()
    );
    let inner_size = quantized.dims[axis + 1..]
        .iter()
        .try_fold(1_usize, |size, dimension| {
            size.checked_mul(usize::try_from(*dimension).ok()?)
        })
        .context("invalid PulseVAD quantized tensor shape")?;

    let float_data = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let parameter_index = if scales.len() == 1 {
                0
            } else {
                (index / inner_size) % axis_size
            };
            let zero_point_index = if zero_points.len() == 1 {
                0
            } else {
                parameter_index
            };
            let centered = value - zero_points[zero_point_index];
            #[allow(clippy::cast_precision_loss)] // Quantized values are bounded integer types.
            let centered = centered as f32;
            centered * scales[parameter_index]
        })
        .collect();
    Ok(TensorProto {
        dims: quantized.dims.clone(),
        data_type: tensor_proto::DataType::Float as i32,
        float_data,
        name: output_name.to_string(),
        ..TensorProto::default()
    })
}

fn integer_values(tensor: &TensorProto) -> Result<Option<Vec<i32>>> {
    let data_type = tensor_proto::DataType::try_from(tensor.data_type)
        .context("invalid PulseVAD tensor data type")?;
    let values = match data_type {
        tensor_proto::DataType::Int8 => {
            raw_or_int32(tensor, |value| i32::from(i8::from_ne_bytes([value])))
        }
        tensor_proto::DataType::Uint8 => raw_or_int32(tensor, i32::from),
        tensor_proto::DataType::Int32 => {
            if tensor.raw_data.is_empty() {
                tensor.int32_data.clone()
            } else {
                let (values, remainder) = tensor.raw_data.as_chunks::<4>();
                anyhow::ensure!(remainder.is_empty(), "invalid PulseVAD INT32 tensor data");
                values
                    .iter()
                    .map(|bytes| i32::from_le_bytes(*bytes))
                    .collect()
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(values))
}

fn raw_or_int32(tensor: &TensorProto, convert: impl Fn(u8) -> i32) -> Vec<i32> {
    if tensor.raw_data.is_empty() {
        tensor.int32_data.clone()
    } else {
        tensor.raw_data.iter().copied().map(convert).collect()
    }
}

fn float_values(tensor: &TensorProto) -> Result<Vec<f32>> {
    anyhow::ensure!(
        tensor.data_type == tensor_proto::DataType::Float as i32,
        "PulseVAD QDQ scale is not FLOAT"
    );
    if tensor.raw_data.is_empty() {
        return Ok(tensor.float_data.clone());
    }
    let (values, remainder) = tensor.raw_data.as_chunks::<4>();
    anyhow::ensure!(remainder.is_empty(), "invalid PulseVAD FLOAT tensor data");
    Ok(values
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Sigmoid saturation and midpoint are exact expectations.
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_stable_for_large_logits() {
        assert_eq!(sigmoid(1_000.0), 1.0);
        assert_eq!(sigmoid(-1_000.0), 0.0);
        assert_eq!(sigmoid(0.0), 0.5);
    }

    #[test]
    fn dequantizes_per_channel_signed_weights() {
        let quantized = TensorProto {
            dims: vec![2, 2],
            data_type: tensor_proto::DataType::Int8 as i32,
            raw_data: vec![u8::MAX, 0, 1, 2],
            ..TensorProto::default()
        };
        let scale = TensorProto {
            dims: vec![2],
            data_type: tensor_proto::DataType::Float as i32,
            float_data: vec![0.5, 2.0],
            ..TensorProto::default()
        };
        let zero_point = TensorProto {
            dims: vec![2],
            data_type: tensor_proto::DataType::Int8 as i32,
            raw_data: vec![0, 1],
            ..TensorProto::default()
        };

        let output =
            dequantize_initializer(&quantized, &scale, Some(&zero_point), 0, "weight").unwrap();

        assert_eq!(output.data_type, tensor_proto::DataType::Float as i32);
        assert_eq!(output.float_data, [-0.5, 0.0, 0.0, 2.0]);
        assert_eq!(output.name, "weight");
    }
}
