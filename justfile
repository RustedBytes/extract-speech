fmt:
    cargo fmt

release: fmt
    cargo build --release

archive:
    ouch compress dist/extract-speech_aarch64-apple-darwin dist/extract-speech_aarch64-apple-darwin.zip
    ouch compress dist/extract-speech_aarch64-unknown-linux-gnu dist/extract-speech_aarch64-unknown-linux-gnu.zip
    ouch compress dist/extract-speech_x86_64-unknown-linux-gnu dist/extract-speech_x86_64-unknown-linux-gnu.zip
