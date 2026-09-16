use anyhow::{Context, Result};
use extract_speech::download::AssetManager;

use super::args::{DownloadArgs, DownloadAsset};

pub(super) fn download(args: &DownloadArgs) -> Result<()> {
    let manager = args
        .cache_dir
        .as_ref()
        .map_or_else(AssetManager::default_cache, |cache_dir| {
            Ok(AssetManager::new(cache_dir))
        })?;

    match args.asset {
        DownloadAsset::All => {
            for files in manager.all_models()? {
                println!("{}", files.model_path().display());
            }
        }
        DownloadAsset::OnnxRuntime => {
            println!("{}", manager.onnx_runtime()?.library_path().display());
        }
        asset => {
            let model_asset = asset
                .model_asset()
                .context("download target does not identify a model asset")?;
            println!("{}", manager.model(model_asset)?.model_path().display());
        }
    }

    Ok(())
}
