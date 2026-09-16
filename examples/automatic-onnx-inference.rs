use extract_speech::{
    audio::load_samples_from_audio_file,
    download::{AssetManager, ModelAsset},
    init_onnx_runtime,
    ort::ep::{ExecutionProviderDispatch, CPU},
    Detector, Error, Model, Result, Runtime, VadParams,
};

fn main() -> Result<()> {
    let audio_path = std::env::args_os()
        .nth(1)
        .ok_or_else(|| Error::msg("usage: automatic-onnx-inference <audio-file>"))?;

    let assets = AssetManager::default_cache()?;
    let bundle = assets.onnx_bundle(ModelAsset::SileroV5)?;
    let providers: Vec<ExecutionProviderDispatch> = vec![CPU::default().build()];
    init_onnx_runtime(bundle.runtime().library_path(), providers.clone())?;

    let samples = load_samples_from_audio_file(audio_path)?;
    let mut detector = Detector::builder(bundle.model().model_path())
        .model(Model::Silero)
        .runtime(Runtime::OnnxRuntime)
        .execution_providers(providers)
        .parameters(VadParams {
            threshold: 0.5,
            ..VadParams::default()
        })
        .build()?;

    for segment in detector.detect(&samples)? {
        println!("{} {}", segment.start, segment.end);
    }
    Ok(())
}
