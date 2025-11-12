default:
    @just --list

build:
    cargo build --workspace

test:
    cargo test --workspace

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

ci: build test clippy

brc:
    cargo run --release -p one-billion-rows --bin create-input
    cargo run --release -p one-billion-rows --bin solve
