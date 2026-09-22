//! Rust scoring core and Python bindings for rouge-rust.
#![forbid(unsafe_code)]

pub mod scorer;

// Unit tests exercise the Rust core without requiring libpython at link time.
// The installed extension is covered separately by the Python integration suite.
#[cfg(not(test))]
mod python;
