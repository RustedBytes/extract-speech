use extract_speech::{
    audio::load_samples_from_audio_file, Detector, Error, Model, Result, Runtime, VadParams,
};

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let model_path = arguments
        .next()
        .ok_or_else(|| Error::msg("usage: library-inference <model.onnx> <audio-file>"))?;
    let audio_path = arguments
        .next()
        .ok_or_else(|| Error::msg("usage: library-inference <model.onnx> <audio-file>"))?;

    let samples = load_samples_from_audio_file(audio_path)?;
    let mut detector = Detector::builder(model_path)
        .model(Model::Silero)
        .runtime(Runtime::Candle)
        .parameters(VadParams {
            threshold: 0.5,
            ..VadParams::default()
        })
        .build()?;

    let segments = detector.detect(&samples)?;
    if segments.is_empty() {
        return Err(Error::msg("the model produced no speech segments"));
    }

    for segment in segments {
        println!("{} {}", segment.start, segment.end);
    }
    Ok(())
}
