//! Candle implementation of `PyAnnote` segmentation.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Context;
use candle_core::Tensor;
use log::debug;
use ndarray::{Array, ArrayD, IxDyn};

use crate::utils;

use candle_onnx::onnx::{attribute_proto, tensor_proto, AttributeProto, NodeProto, TensorProto};

pub struct PyAnnote {
    model: candle_onnx::onnx::ModelProto,
    device: candle_core::Device,
}

impl PyAnnote {
    /// Loads a `PyAnnote` model for Candle inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX model cannot be read.
    pub fn new(
        _vad_params: utils::VadParams,
        model_path: PathBuf,
        device: candle_core::Device,
    ) -> anyhow::Result<Self> {
        let mut model = candle_onnx::read_file(model_path)?;
        expand_instance_normalization(&mut model)?;
        expand_1d_max_pool(&mut model)?;
        expand_bidirectional_lstm(&mut model)?;
        expand_lstm_output_reshape(&mut model)?;
        Ok(Self { model, device })
    }

    pub fn reset(&mut self) {
        // PyAnnote does not maintain state between calls.
    }

    /// Computes frame-level segmentation logits.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails or the output tensor is invalid.
    pub fn get_frame_probabilities(
        &mut self,
        audio_samples: &[f32],
    ) -> anyhow::Result<ArrayD<f32>> {
        let input = Tensor::from_slice(audio_samples, (1, 1, audio_samples.len()), &self.device)?;
        debug!(
            "PyAnnote input: {:?}, dtype: {:?}",
            input.shape(),
            input.dtype()
        );

        let outputs = candle_onnx::simple_eval(
            &self.model,
            HashMap::from_iter([("input_values".to_string(), input)]),
        )
        .context("failed to evaluate PyAnnote with Candle")?;
        let logits = outputs
            .get("logits")
            .context("PyAnnote model did not return a 'logits' tensor")?;
        let dimensions = logits.dims();
        anyhow::ensure!(
            dimensions.len() == 3,
            "PyAnnote returned logits with shape {dimensions:?}; expected [batch, frames, classes]"
        );
        debug!("PyAnnote output shape: {dimensions:?}");

        let values = logits.flatten_all()?.to_vec1::<f32>()?;
        Array::from_shape_vec(IxDyn(dimensions), values).context("invalid PyAnnote output tensor")
    }
}

fn expand_lstm_output_reshape(model: &mut candle_onnx::onnx::ModelProto) -> anyhow::Result<()> {
    let graph = model
        .graph
        .as_mut()
        .context("PyAnnote ONNX model does not contain a graph")?;
    let mut nodes = Vec::with_capacity(graph.node.len());

    for (index, node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
        if node.op_type != "Reshape" || !node.name.starts_with("/lstm/Reshape") {
            nodes.push(node);
            continue;
        }
        anyhow::ensure!(
            node.input.len() == 2 && node.output.len() == 1,
            "invalid PyAnnote LSTM output Reshape node"
        );
        let flattened = format!("__pyannote_lstm_reshape_{index}_flattened");
        nodes.push(node_with_attributes(
            "Flatten",
            &[&node.input[0]],
            &[&flattened],
            vec![int_attribute("axis", 2)],
        ));
        nodes.push(node_with_attributes(
            "Unsqueeze",
            &[&flattened],
            &[&node.output[0]],
            vec![ints_attribute("axes", vec![1])],
        ));
    }

    graph.node = nodes;
    Ok(())
}

#[allow(clippy::too_many_lines)] // The rewrite keeps the complete bidirectional LSTM contract together.
fn expand_bidirectional_lstm(model: &mut candle_onnx::onnx::ModelProto) -> anyhow::Result<()> {
    let graph = model
        .graph
        .as_mut()
        .context("PyAnnote ONNX model does not contain a graph")?;
    let mut nodes = Vec::with_capacity(graph.node.len());

    for (index, mut node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
        let is_bidirectional = node.op_type == "LSTM"
            && node
                .attribute
                .iter()
                .any(|attribute| attribute.name == "direction" && attribute.s == b"bidirectional");
        if !is_bidirectional {
            nodes.push(node);
            continue;
        }
        anyhow::ensure!(
            node.input.len() >= 3 && !node.output.is_empty(),
            "invalid PyAnnote bidirectional LSTM node"
        );

        let prefix = format!("__pyannote_lstm_{index}");
        let forward_index = format!("{prefix}_forward_index");
        let backward_index = format!("{prefix}_backward_index");
        let reverse_starts = format!("{prefix}_reverse_starts");
        let reverse_ends = format!("{prefix}_reverse_ends");
        let reverse_axes = format!("{prefix}_reverse_axes");
        let reverse_steps = format!("{prefix}_reverse_steps");
        graph
            .initializer
            .push(int64_initializer(&forward_index, &[0]));
        graph
            .initializer
            .push(int64_initializer(&backward_index, &[1]));
        graph
            .initializer
            .push(int64_initializer(&reverse_starts, &[-1]));
        graph
            .initializer
            .push(int64_initializer(&reverse_ends, &[i64::MIN]));
        graph
            .initializer
            .push(int64_initializer(&reverse_axes, &[0]));
        graph
            .initializer
            .push(int64_initializer(&reverse_steps, &[-1]));

        let reversed_input = format!("{prefix}_reversed_input");
        nodes.push(node_proto(
            "Slice",
            &[
                &node.input[0],
                &reverse_starts,
                &reverse_ends,
                &reverse_axes,
                &reverse_steps,
            ],
            &[&reversed_input],
        ));

        let mut forward_inputs = node.input.clone();
        let mut backward_inputs = node.input.clone();
        backward_inputs[0].clone_from(&reversed_input);
        for input_index in [1_usize, 2, 3, 5, 6, 7] {
            let Some(input) = node
                .input
                .get(input_index)
                .filter(|input| !input.is_empty())
            else {
                continue;
            };
            let forward = format!("{prefix}_input_{input_index}_forward");
            let backward = format!("{prefix}_input_{input_index}_backward");
            nodes.push(node_with_attributes(
                "Gather",
                &[input, &forward_index],
                &[&forward],
                vec![int_attribute("axis", 0)],
            ));
            nodes.push(node_with_attributes(
                "Gather",
                &[input, &backward_index],
                &[&backward],
                vec![int_attribute("axis", 0)],
            ));
            forward_inputs[input_index] = forward;
            backward_inputs[input_index] = backward;
        }

        for attribute in &mut node.attribute {
            if attribute.name == "direction" {
                attribute.s = b"forward".to_vec();
            } else if attribute.name == "activations" && attribute.strings.len() == 6 {
                attribute.strings.truncate(3);
            }
        }

        let original_outputs = node.output.clone();
        let forward_outputs: Vec<String> = original_outputs
            .iter()
            .enumerate()
            .map(|(output_index, output)| {
                if output.is_empty() {
                    String::new()
                } else {
                    format!("{prefix}_output_{output_index}_forward")
                }
            })
            .collect();
        let backward_outputs: Vec<String> = original_outputs
            .iter()
            .enumerate()
            .map(|(output_index, output)| {
                if output.is_empty() {
                    String::new()
                } else {
                    format!("{prefix}_output_{output_index}_backward")
                }
            })
            .collect();

        let mut forward_node = node.clone();
        forward_node.input = forward_inputs;
        forward_node.output.clone_from(&forward_outputs);
        nodes.push(forward_node);
        node.input = backward_inputs;
        node.output.clone_from(&backward_outputs);
        nodes.push(node);

        for (output_index, original_output) in original_outputs.iter().enumerate() {
            if original_output.is_empty() {
                continue;
            }
            let forward = &forward_outputs[output_index];
            let mut backward = backward_outputs[output_index].clone();
            if output_index == 0 {
                let reversed_output = format!("{prefix}_output_0_backward_reversed");
                nodes.push(node_proto(
                    "Slice",
                    &[
                        &backward,
                        &reverse_starts,
                        &reverse_ends,
                        &reverse_axes,
                        &reverse_steps,
                    ],
                    &[&reversed_output],
                ));
                backward = reversed_output;
            }
            nodes.push(node_with_attributes(
                "Concat",
                &[forward, &backward],
                &[original_output],
                vec![int_attribute("axis", i64::from(output_index == 0))],
            ));
        }
    }

    graph.node = nodes;
    Ok(())
}

fn expand_1d_max_pool(model: &mut candle_onnx::onnx::ModelProto) -> anyhow::Result<()> {
    let graph = model
        .graph
        .as_mut()
        .context("PyAnnote ONNX model does not contain a graph")?;
    let mut nodes = Vec::with_capacity(graph.node.len());

    for (index, mut node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
        let is_1d = node.op_type == "MaxPool"
            && node
                .attribute
                .iter()
                .find(|attribute| attribute.name == "kernel_shape")
                .is_some_and(|attribute| attribute.ints.len() == 1);
        if !is_1d {
            nodes.push(node);
            continue;
        }
        anyhow::ensure!(
            node.input.len() == 1 && node.output.len() == 1,
            "invalid PyAnnote MaxPool node"
        );

        let prefix = format!("__pyannote_max_pool_{index}");
        let expanded = format!("{prefix}_expanded");
        let pooled = format!("{prefix}_pooled");
        let output = node.output[0].clone();
        nodes.push(node_with_attributes(
            "Unsqueeze",
            &[&node.input[0]],
            &[&expanded],
            vec![ints_attribute("axes", vec![2])],
        ));

        for attribute in &mut node.attribute {
            match attribute.name.as_str() {
                "kernel_shape" | "strides" | "dilations" => attribute.ints.insert(0, 1),
                "pads" => {
                    anyhow::ensure!(
                        attribute.ints.len() == 2,
                        "invalid padding on PyAnnote MaxPool node"
                    );
                    attribute.ints.insert(0, 0);
                    attribute.ints.insert(2, 0);
                }
                _ => {}
            }
        }
        node.input[0].clone_from(&expanded);
        node.output[0].clone_from(&pooled);
        nodes.push(node);
        nodes.push(node_with_attributes(
            "Squeeze",
            &[&pooled],
            &[&output],
            vec![ints_attribute("axes", vec![2])],
        ));
    }

    graph.node = nodes;
    Ok(())
}

fn expand_instance_normalization(model: &mut candle_onnx::onnx::ModelProto) -> anyhow::Result<()> {
    let graph = model
        .graph
        .as_mut()
        .context("PyAnnote ONNX model does not contain a graph")?;
    let mut nodes = Vec::with_capacity(graph.node.len());

    for (index, node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
        if node.op_type != "InstanceNormalization" {
            nodes.push(node);
            continue;
        }
        anyhow::ensure!(
            node.input.len() == 3 && node.output.len() == 1,
            "invalid PyAnnote InstanceNormalization node"
        );

        let epsilon = node
            .attribute
            .iter()
            .find(|attribute| attribute.name == "epsilon")
            .map_or(1e-5, |attribute| attribute.f);
        let prefix = format!("__pyannote_instance_norm_{index}");
        let epsilon_name = format!("{prefix}_epsilon");
        let channel_shape_name = format!("{prefix}_channel_shape");
        graph
            .initializer
            .push(float_initializer(&epsilon_name, epsilon));
        graph
            .initializer
            .push(int64_initializer(&channel_shape_name, &[1, -1, 1]));

        let mean = format!("{prefix}_mean");
        let centered = format!("{prefix}_centered");
        let squared = format!("{prefix}_squared");
        let variance = format!("{prefix}_variance");
        let adjusted_variance = format!("{prefix}_adjusted_variance");
        let deviation = format!("{prefix}_deviation");
        let normalized = format!("{prefix}_normalized");
        let scale = format!("{prefix}_scale");
        let bias = format!("{prefix}_bias");
        let scaled = format!("{prefix}_scaled");

        nodes.push(node_with_attributes(
            "ReduceMean",
            &[&node.input[0]],
            &[&mean],
            vec![
                ints_attribute("axes", vec![2]),
                int_attribute("keepdims", 1),
            ],
        ));
        nodes.push(node_proto("Sub", &[&node.input[0], &mean], &[&centered]));
        nodes.push(node_proto("Mul", &[&centered, &centered], &[&squared]));
        nodes.push(node_with_attributes(
            "ReduceMean",
            &[&squared],
            &[&variance],
            vec![
                ints_attribute("axes", vec![2]),
                int_attribute("keepdims", 1),
            ],
        ));
        nodes.push(node_proto(
            "Add",
            &[&variance, &epsilon_name],
            &[&adjusted_variance],
        ));
        nodes.push(node_proto("Sqrt", &[&adjusted_variance], &[&deviation]));
        nodes.push(node_proto("Div", &[&centered, &deviation], &[&normalized]));
        nodes.push(node_proto(
            "Reshape",
            &[&node.input[1], &channel_shape_name],
            &[&scale],
        ));
        nodes.push(node_proto(
            "Reshape",
            &[&node.input[2], &channel_shape_name],
            &[&bias],
        ));
        nodes.push(node_proto("Mul", &[&normalized, &scale], &[&scaled]));
        nodes.push(node_proto("Add", &[&scaled, &bias], &[&node.output[0]]));
    }

    graph.node = nodes;
    Ok(())
}

fn node_proto(op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    node_with_attributes(op_type, inputs, outputs, Vec::new())
}

fn node_with_attributes(
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attribute: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|value| (*value).to_string()).collect(),
        output: outputs.iter().map(|value| (*value).to_string()).collect(),
        op_type: op_type.to_string(),
        attribute,
        ..NodeProto::default()
    }
}

fn ints_attribute(name: &str, ints: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Ints as i32,
        ints,
        ..AttributeProto::default()
    }
}

fn int_attribute(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Int as i32,
        i: value,
        ..AttributeProto::default()
    }
}

fn float_initializer(name: &str, value: f32) -> TensorProto {
    TensorProto {
        data_type: tensor_proto::DataType::Float as i32,
        float_data: vec![value],
        name: name.to_string(),
        ..TensorProto::default()
    }
}

fn int64_initializer(name: &str, values: &[i64]) -> TensorProto {
    TensorProto {
        dims: vec![i64::try_from(values.len()).expect("initializer length fits in i64")],
        data_type: tensor_proto::DataType::Int64 as i32,
        int64_data: values.to_vec(),
        name: name.to_string(),
        ..TensorProto::default()
    }
}
