fmt:
    cargo fmt

clippy:
    cargo clippy --all-targets -- -D warnings

release: fmt
    cargo build --release

test:
    cargo test
