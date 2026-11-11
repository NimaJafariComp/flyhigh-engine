#!/usr/bin/env sh
# Build the browser module into examples/web/pkg (not in git): the wasm32 crate with the size-first `wasm` profile (opt-level z,
# fat LTO: 1.02 MB), then wasm-bindgen with the CLI version that matches Cargo.lock (0.2.129), and the browser adapters
# (crates/flyhigh-web/js/adapters.js) copied next to the module. Run from
# the repository root; the demo pages in web/ and the baseline recorder use it.
set -e
cd "$(dirname "$0")/.."
# panic locations carry source paths: without the remap the module ships the builder's home directory
cargo_home=${CARGO_HOME:-$HOME/.cargo}
root=$(pwd)
if command -v cygpath >/dev/null 2>&1; then
  cargo_home=$(cygpath -w "$cargo_home")
  root=$(cygpath -w "$root")
fi
sep=$(printf '\037')
export CARGO_ENCODED_RUSTFLAGS="--remap-path-prefix=$cargo_home=cargo$sep--remap-path-prefix=$root=flyhigh"
cargo build -p flyhigh-web --profile wasm --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir examples/web/pkg target/wasm32-unknown-unknown/wasm/flyhigh_web.wasm
cp crates/flyhigh-web/js/adapters.js examples/web/pkg/
ls -l examples/web/pkg/flyhigh_web_bg.wasm
