//! Checksum-verified downloads and persistent caching for model bundles and
//! the dynamically loaded ONNX Runtime library.

use std::{
    env,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};

/// ONNX Runtime version compatible with this crate's `ort` dependency.
pub const ONNX_RUNTIME_VERSION: &str = "1.27.1";

/// A revision-pinned model artifact known to the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelAsset {
    /// Official Silero VAD v6 ONNX graph.
    SileroV6,
    SileroV5,
    PyAnnoteSegmentation,
    PulseVadFp32,
    PulseVadInt8,
    FsmnVadFp32,
    FsmnVadInt8,
    TenVad,
    MarbleNetFp32,
    MarbleNetInt8,
}

impl ModelAsset {
    /// Every model asset currently supported by [`AssetManager::model`].
    pub const ALL: [Self; 10] = [
        Self::SileroV6,
        Self::SileroV5,
        Self::PyAnnoteSegmentation,
        Self::PulseVadFp32,
        Self::PulseVadInt8,
        Self::FsmnVadFp32,
        Self::FsmnVadInt8,
        Self::TenVad,
        Self::MarbleNetFp32,
        Self::MarbleNetInt8,
    ];

    fn spec(self) -> ModelSpec {
        match self {
            Self::SileroV6 => ModelSpec {
                directory: "silero-v6",
                model_file: "silero_vad.onnx",
                artifacts: &SILERO_V6_ARTIFACTS,
            },
            Self::SileroV5 => ModelSpec {
                directory: "silero-v5",
                model_file: "model.onnx",
                artifacts: &SILERO_V5_ARTIFACTS,
            },
            Self::PyAnnoteSegmentation => ModelSpec {
                directory: "pyannote-segmentation-3.0",
                model_file: "model.onnx",
                artifacts: &PYANNOTE_ARTIFACTS,
            },
            Self::PulseVadFp32 => ModelSpec {
                directory: "pulsevad-fp32",
                model_file: "pulsevad_2.1k.onnx",
                artifacts: &PULSEVAD_FP32_ARTIFACTS,
            },
            Self::PulseVadInt8 => ModelSpec {
                directory: "pulsevad-int8",
                model_file: "pulsevad_2.1k_int8.onnx",
                artifacts: &PULSEVAD_INT8_ARTIFACTS,
            },
            Self::FsmnVadFp32 => ModelSpec {
                directory: "fsmn-vad-fp32",
                model_file: "model.onnx",
                artifacts: &FSMN_FP32_ARTIFACTS,
            },
            Self::FsmnVadInt8 => ModelSpec {
                directory: "fsmn-vad-int8",
                model_file: "model_quant.onnx",
                artifacts: &FSMN_INT8_ARTIFACTS,
            },
            Self::TenVad => ModelSpec {
                directory: "ten-vad",
                model_file: "ten-vad.onnx",
                artifacts: &TEN_ARTIFACTS,
            },
            Self::MarbleNetFp32 => ModelSpec {
                directory: "marblenet-fp32",
                model_file: "marblenet.onnx",
                artifacts: &MARBLENET_FP32_ARTIFACTS,
            },
            Self::MarbleNetInt8 => ModelSpec {
                directory: "marblenet-int8",
                model_file: "marblenet_int8.onnx",
                artifacts: &MARBLENET_INT8_ARTIFACTS,
            },
        }
    }
}

/// Paths returned after a complete model bundle is available in the cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFiles {
    asset: ModelAsset,
    directory: PathBuf,
    model_path: PathBuf,
    files: Vec<PathBuf>,
}

impl ModelFiles {
    #[must_use]
    pub fn asset(&self) -> ModelAsset {
        self.asset
    }

    /// Main ONNX graph to pass to `Detector::builder`.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Directory containing the model and all required sidecar files.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Every file in the downloaded bundle, including the main model.
    #[must_use]
    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }
}

/// Paths returned after ONNX Runtime is downloaded and extracted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxRuntimeFiles {
    archive_path: PathBuf,
    library_path: PathBuf,
}

/// A complete ONNX inference bundle containing a model (and its sidecars) plus
/// the host's compatible ONNX Runtime distribution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxBundle {
    model: ModelFiles,
    runtime: OnnxRuntimeFiles,
}

impl OnnxBundle {
    #[must_use]
    pub fn model(&self) -> &ModelFiles {
        &self.model
    }

    #[must_use]
    pub fn runtime(&self) -> &OnnxRuntimeFiles {
        &self.runtime
    }
}

impl OnnxRuntimeFiles {
    #[must_use]
    pub fn version(&self) -> &'static str {
        ONNX_RUNTIME_VERSION
    }

    #[must_use]
    pub fn archive_path(&self) -> &Path {
        &self.archive_path
    }

    /// Dynamic library to pass to `init_onnx_runtime`.
    #[must_use]
    pub fn library_path(&self) -> &Path {
        &self.library_path
    }
}

/// Downloads model bundles and runtime files into a persistent cache.
#[derive(Clone, Debug)]
pub struct AssetManager {
    cache_dir: PathBuf,
}

impl AssetManager {
    /// Uses an explicit cache directory.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    /// Uses `EXTRACT_SPEECH_CACHE_DIR` or the platform's conventional cache directory.
    ///
    /// # Errors
    ///
    /// Returns an error when no platform cache directory can be resolved.
    pub fn default_cache() -> Result<Self> {
        Ok(Self::new(default_cache_dir()?))
    }

    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Ensures the selected model and every required sidecar file are cached.
    ///
    /// # Errors
    ///
    /// Returns an error if a cache directory cannot be created, an artifact
    /// cannot be downloaded, or its checksum does not match the registry.
    pub fn model(&self, asset: ModelAsset) -> Result<ModelFiles> {
        let spec = asset.spec();
        let directory = self.cache_dir.join("models").join(spec.directory);
        fs::create_dir_all(&directory)
            .with_context(|| format!("failed to create cache directory {}", directory.display()))?;

        let mut files = Vec::with_capacity(spec.artifacts.len());
        for artifact in spec.artifacts {
            let destination = directory.join(artifact.filename);
            download_verified(artifact.url, artifact.sha256, &destination)?;
            files.push(destination);
        }

        Ok(ModelFiles {
            asset,
            model_path: directory.join(spec.model_file),
            directory,
            files,
        })
    }

    /// Downloads every supported model bundle. Existing valid files are reused.
    ///
    /// # Errors
    ///
    /// Returns the first model download, filesystem, or checksum error.
    pub fn all_models(&self) -> Result<Vec<ModelFiles>> {
        ModelAsset::ALL
            .into_iter()
            .map(|asset| self.model(asset))
            .collect()
    }

    /// Prepares a model bundle and ONNX Runtime with one call.
    ///
    /// # Errors
    ///
    /// Returns an error if either the model or runtime bundle cannot be cached.
    pub fn onnx_bundle(&self, asset: ModelAsset) -> Result<OnnxBundle> {
        Ok(OnnxBundle {
            model: self.model(asset)?,
            runtime: self.onnx_runtime()?,
        })
    }

    /// Downloads and extracts a CPU ONNX Runtime distribution for the host.
    ///
    /// Supported hosts are Linux x86-64/ARM64, macOS ARM64, and Windows
    /// x86-64/ARM64.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported hosts or when the runtime cannot be
    /// downloaded, verified, or extracted.
    pub fn onnx_runtime(&self) -> Result<OnnxRuntimeFiles> {
        let spec = runtime_spec()?;
        let directory = self
            .cache_dir
            .join("onnxruntime")
            .join(ONNX_RUNTIME_VERSION)
            .join(spec.target);
        fs::create_dir_all(&directory)
            .with_context(|| format!("failed to create cache directory {}", directory.display()))?;

        let archive_path = directory.join(spec.archive_name);
        download_verified(spec.url, spec.sha256, &archive_path)?;

        let library_path = directory.join(spec.library_name);
        let checksum_path = directory.join(format!("{}.sha256", spec.library_name));
        let library_is_valid = library_path.is_file()
            && checksum_path.is_file()
            && fs::read_to_string(&checksum_path).is_ok_and(|expected| {
                file_sha256(&library_path).is_ok_and(|actual| actual == expected.trim())
            });
        if !library_is_valid {
            if library_path.exists() {
                fs::remove_file(&library_path).with_context(|| {
                    format!(
                        "failed to remove invalid runtime library {}",
                        library_path.display()
                    )
                })?;
            }
            extract_runtime_library(&archive_path, &library_path, spec.format)?;
            fs::write(&checksum_path, file_sha256(&library_path)?)
                .with_context(|| format!("failed to write {}", checksum_path.display()))?;
        }

        Ok(OnnxRuntimeFiles {
            archive_path,
            library_path,
        })
    }
}

/// Resolves the persistent cache directory without creating it.
///
/// # Errors
///
/// Returns an error when the required platform environment variables are absent.
pub fn default_cache_dir() -> Result<PathBuf> {
    if let Some(path) = env::var_os("EXTRACT_SPEECH_CACHE_DIR").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }

    if cfg!(target_os = "windows") {
        return env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .map(|path| path.join("extract-speech"))
            .context("LOCALAPPDATA is not set; set EXTRACT_SPEECH_CACHE_DIR explicitly");
    }

    let home = env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set; set EXTRACT_SPEECH_CACHE_DIR explicitly")?;
    if cfg!(target_os = "macos") {
        Ok(home.join("Library").join("Caches").join("extract-speech"))
    } else if let Some(path) = env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        Ok(PathBuf::from(path).join("extract-speech"))
    } else {
        Ok(home.join(".cache").join("extract-speech"))
    }
}

#[derive(Clone, Copy)]
struct ArtifactSpec {
    filename: &'static str,
    url: &'static str,
    sha256: &'static str,
}

struct ModelSpec {
    directory: &'static str,
    model_file: &'static str,
    artifacts: &'static [ArtifactSpec],
}

const SILERO_V6_ARTIFACTS: [ArtifactSpec; 1] = [ArtifactSpec {
    filename: "silero_vad.onnx",
    url: "https://raw.githubusercontent.com/snakers4/silero-vad/60b7ffa243625ebdc1070275a29f18c87843786a/src/silero_vad/data/silero_vad.onnx",
    sha256: "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
}];
const SILERO_V5_ARTIFACTS: [ArtifactSpec; 1] = [ArtifactSpec {
    filename: "model.onnx",
    url: "https://huggingface.co/onnx-community/silero-vad/resolve/ddc9a7e80d6758f6fc795a1e8a04b798eb929d3a/onnx/model.onnx",
    sha256: "a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808",
}];
const PYANNOTE_ARTIFACTS: [ArtifactSpec; 1] = [ArtifactSpec {
    filename: "model.onnx",
    url: "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/733a93b6473d019a773298e08cefa686894b1854/onnx/model.onnx",
    sha256: "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25",
}];
const PULSEVAD_FP32_ARTIFACTS: [ArtifactSpec; 1] = [ArtifactSpec {
    filename: "pulsevad_2.1k.onnx",
    url: "https://raw.githubusercontent.com/AydinAdnan/PulseVAD/af25e79d66830a3fee74541812721f6158fc92b5/pulsevad/data/pulsevad_2.1k.onnx",
    sha256: "2b8c4874fc4ecd64916fc8726e2a8281b1cb9457c21f42a23a9776a4d538c665",
}];
const PULSEVAD_INT8_ARTIFACTS: [ArtifactSpec; 1] = [ArtifactSpec {
    filename: "pulsevad_2.1k_int8.onnx",
    url: "https://raw.githubusercontent.com/AydinAdnan/PulseVAD/af25e79d66830a3fee74541812721f6158fc92b5/pulsevad/data/pulsevad_2.1k_int8.onnx",
    sha256: "416061347a1e723ed15163acd51006bf3c513b27bb9f57d85e2c694cc44b8389",
}];
const FSMN_FP32_ARTIFACTS: [ArtifactSpec; 2] = [
    ArtifactSpec {
        filename: "model.onnx",
        url: "https://huggingface.co/funasr/fsmn-vad-onnx/resolve/f6e9fbb4cefa7397216c763f21307993f147f585/model.onnx",
        sha256: "756887ce01695a9bb00dd85ca0f743653de03b18ba54d2e9ef4f4bb9b3edbf9f",
    },
    ArtifactSpec {
        filename: "vad.mvn",
        url: "https://huggingface.co/funasr/fsmn-vad-onnx/resolve/f6e9fbb4cefa7397216c763f21307993f147f585/vad.mvn",
        sha256: "6820fef9687708c4fc3fab2530179c8fcea6262daa25514380056cd8f6eb1754",
    },
];
const FSMN_INT8_ARTIFACTS: [ArtifactSpec; 2] = [
    ArtifactSpec {
        filename: "model_quant.onnx",
        url: "https://huggingface.co/funasr/fsmn-vad-onnx/resolve/f6e9fbb4cefa7397216c763f21307993f147f585/model_quant.onnx",
        sha256: "9b28837838fce9685503c63139fadbad35d6c8ed485485dafdbb32e725969660",
    },
    ArtifactSpec {
        filename: "vad.mvn",
        url: "https://huggingface.co/funasr/fsmn-vad-onnx/resolve/f6e9fbb4cefa7397216c763f21307993f147f585/vad.mvn",
        sha256: "6820fef9687708c4fc3fab2530179c8fcea6262daa25514380056cd8f6eb1754",
    },
];
const TEN_ARTIFACTS: [ArtifactSpec; 1] = [ArtifactSpec {
    filename: "ten-vad.onnx",
    url: "https://huggingface.co/TEN-framework/ten-vad/resolve/bda8ffc78b1846c5c7cbd38f04e52deff49de707/src/onnx_model/ten-vad.onnx",
    sha256: "e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4",
}];
const MARBLENET_FP32_ARTIFACTS: [ArtifactSpec; 1] = [ArtifactSpec {
    filename: "marblenet.onnx",
    url: "https://huggingface.co/TigreGotico/frame-vad-marblenet-onnx/resolve/e8786fe74e055954901eb553cc9c3145323981cc/marblenet.onnx",
    sha256: "4ad3364be94d462b5fd4fa39910c24967dbb9dba436e27bcff7a88359515e491",
}];
const MARBLENET_INT8_ARTIFACTS: [ArtifactSpec; 1] = [ArtifactSpec {
    filename: "marblenet_int8.onnx",
    url: "https://huggingface.co/TigreGotico/frame-vad-marblenet-onnx/resolve/e8786fe74e055954901eb553cc9c3145323981cc/marblenet_int8.onnx",
    sha256: "9c4462323f9b576fd5e581d3c86b9b9b513468d18a79bcdcd3a2bcbcaab02699",
}];

fn download_verified(url: &str, expected_sha256: &str, destination: &Path) -> Result<()> {
    if destination.is_file() && file_sha256(destination)? == expected_sha256 {
        return Ok(());
    }

    if destination.exists() {
        fs::remove_file(destination).with_context(|| {
            format!(
                "failed to remove invalid cache file {}",
                destination.display()
            )
        })?;
    }
    let parent = destination
        .parent()
        .context("cache destination has no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create cache directory {}", parent.display()))?;

    let mut response = ureq::get(url)
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to create a temporary file in {}", parent.display()))?;
    let mut reader = response.body_mut().as_reader();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("failed while downloading {url}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        temporary
            .write_all(&buffer[..read])
            .with_context(|| format!("failed to write {}", destination.display()))?;
    }
    temporary.as_file_mut().sync_all()?;

    let actual_sha256 = encode_hex(hasher.finalize().as_ref());
    if actual_sha256 != expected_sha256 {
        bail!("checksum mismatch for {url}: expected {expected_sha256}, received {actual_sha256}");
    }

    match temporary.persist(destination) {
        Ok(_) => Ok(()),
        Err(_error)
            if destination.is_file()
                && file_sha256(destination).ok().as_deref() == Some(expected_sha256) =>
        {
            Ok(())
        }
        Err(error) => {
            Err(error.error).with_context(|| format!("failed to cache {}", destination.display()))
        }
    }
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open cached file {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read cached file {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(encode_hex(hasher.finalize().as_ref()))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[derive(Clone, Copy)]
enum ArchiveFormat {
    TarGz,
    Zip,
}

struct RuntimeSpec {
    target: &'static str,
    archive_name: &'static str,
    library_name: &'static str,
    url: &'static str,
    sha256: &'static str,
    format: ArchiveFormat,
}

fn runtime_spec() -> Result<RuntimeSpec> {
    let spec = match (env::consts::OS, env::consts::ARCH) {
        ("linux", "x86_64") => RuntimeSpec {
            target: "linux-x64",
            archive_name: "onnxruntime-linux-x64-1.27.1.tgz",
            library_name: "libonnxruntime.so",
            url: "https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-linux-x64-1.27.1.tgz",
            sha256: "25b1ef1fea1acd210d63f8f24dc870ad6e077795ce1f54876252c6d3803c15af",
            format: ArchiveFormat::TarGz,
        },
        ("linux", "aarch64") => RuntimeSpec {
            target: "linux-aarch64",
            archive_name: "onnxruntime-linux-aarch64-1.27.1.tgz",
            library_name: "libonnxruntime.so",
            url: "https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-linux-aarch64-1.27.1.tgz",
            sha256: "33c67e33d1e25b816878366ea276589a024f71f000e7ff955c4b33224d639edd",
            format: ArchiveFormat::TarGz,
        },
        ("macos", "aarch64") => RuntimeSpec {
            target: "macos-arm64",
            archive_name: "onnxruntime-osx-arm64-1.27.1.tgz",
            library_name: "libonnxruntime.dylib",
            url: "https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-osx-arm64-1.27.1.tgz",
            sha256: "e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6",
            format: ArchiveFormat::TarGz,
        },
        ("windows", "x86_64") => RuntimeSpec {
            target: "windows-x64",
            archive_name: "onnxruntime-win-x64-1.27.1.zip",
            library_name: "onnxruntime.dll",
            url: "https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-win-x64-1.27.1.zip",
            sha256: "2e00414a63fdef0914cd5a5ede6c707844878e0c08e1b6693842f0451b2df2a1",
            format: ArchiveFormat::Zip,
        },
        ("windows", "aarch64") => RuntimeSpec {
            target: "windows-arm64",
            archive_name: "onnxruntime-win-arm64-1.27.1.zip",
            library_name: "onnxruntime.dll",
            url: "https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-win-arm64-1.27.1.zip",
            sha256: "6e22c2061ba6400b42a59663d700c8694e4e8fe654cf452c4700c24237407ae1",
            format: ArchiveFormat::Zip,
        },
        (os, arch) => bail!("automatic ONNX Runtime download is not supported on {os}/{arch}"),
    };
    Ok(spec)
}

fn extract_runtime_library(
    archive_path: &Path,
    destination: &Path,
    format: ArchiveFormat,
) -> Result<()> {
    let parent = destination
        .parent()
        .context("runtime library destination has no parent directory")?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;

    match format {
        ArchiveFormat::TarGz => {
            let archive = File::open(archive_path)?;
            let mut archive = tar::Archive::new(GzDecoder::new(archive));
            let mut found = false;
            for entry in archive.entries()? {
                let mut entry = entry?;
                let path = entry.path()?;
                let name = path.file_name().and_then(|name| name.to_str());
                let matches = destination
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|expected| {
                        name == Some(expected)
                            || (expected == "libonnxruntime.so"
                                && name.is_some_and(|name| name.starts_with("libonnxruntime.so.")))
                            || (expected == "libonnxruntime.dylib"
                                && name.is_some_and(|name| {
                                    name.starts_with("libonnxruntime.")
                                        && Path::new(name).extension().is_some_and(|extension| {
                                            extension.eq_ignore_ascii_case("dylib")
                                        })
                                }))
                    });
                if matches && entry.header().entry_type().is_file() {
                    std::io::copy(&mut entry, temporary.as_file_mut())?;
                    found = true;
                    break;
                }
            }
            if !found {
                bail!("ONNX Runtime archive does not contain the expected dynamic library");
            }
        }
        ArchiveFormat::Zip => {
            let archive = File::open(archive_path)?;
            let mut archive = zip::ZipArchive::new(archive)?;
            let mut found = false;
            for index in 0..archive.len() {
                let mut entry = archive.by_index(index)?;
                if Path::new(entry.name()).file_name() == destination.file_name() {
                    std::io::copy(&mut entry, temporary.as_file_mut())?;
                    found = true;
                    break;
                }
            }
            if !found {
                bail!("ONNX Runtime archive does not contain the expected dynamic library");
            }
        }
    }

    temporary.as_file_mut().sync_all()?;
    temporary
        .persist(destination)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to extract {}", destination.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use super::*;

    #[test]
    fn registry_contains_unique_complete_model_bundles() {
        let mut directories = std::collections::HashSet::new();
        for asset in ModelAsset::ALL {
            let spec = asset.spec();
            assert!(directories.insert(spec.directory));
            assert!(spec
                .artifacts
                .iter()
                .any(|file| file.filename == spec.model_file));
            for artifact in spec.artifacts {
                assert_eq!(artifact.sha256.len(), 64);
                assert!(artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
                assert!(artifact.url.starts_with("https://"));
            }
        }
    }

    #[test]
    fn verified_download_is_reused_from_cache() {
        let payload = b"cached model bytes";
        let expected = encode_hex(Sha256::digest(payload).as_ref());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            )
            .unwrap();
            stream.write_all(payload).unwrap();
        });

        let cache = tempfile::tempdir().unwrap();
        let destination = cache.path().join("model.onnx");
        let url = format!("http://{address}/model.onnx");
        download_verified(&url, &expected, &destination).unwrap();
        server.join().unwrap();

        download_verified("http://127.0.0.1:1/unreachable", &expected, &destination).unwrap();
        assert_eq!(fs::read(destination).unwrap(), payload);
    }
}
