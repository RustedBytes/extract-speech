fmt:
    cargo fmt

release: fmt
    cargo build --release

archive:
    ouch compress dist/... ...
