fmt:
    cargo fmt

clippy:
    cargo clippy --all-targets --all-features -- -D warnings

release: fmt
    cargo build --release --features cli

test:
    cargo test --all-targets --all-features

test-models:
    ./scripts/test-models.sh
