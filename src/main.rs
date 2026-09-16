#[cfg(all(feature = "accelerate-src", target_vendor = "apple"))]
extern crate accelerate_src;

mod cli;

fn main() -> anyhow::Result<()> {
    cli::run()
}
