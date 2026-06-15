//! `xtask` — integral maintainer tooling.
//!
//! Subcommands:
//!
//! - `gen-rys-tables` — regenerate `crates/integral/src/math/rys_tables.rs` from the
//!   in-repo discretized-Stieltjes reference (pure Rust, no external library). A
//!   unit test re-checks the committed table against the live reference so it
//!   cannot drift.
//! - `release <X.Y.Z>` — bump the workspace version and stamp the changelog.
//! - `help` — print usage.
//!
//! This crate is intentionally tiny and dependency-light. It is not published.

mod gen_rys_tables;
mod release;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "gen-rys-tables" => {
            gen_rys_tables::run();
            ExitCode::SUCCESS
        }
        "release" => match args.get(1) {
            Some(version) => match release::run(version) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("xtask release: {e}");
                    ExitCode::FAILURE
                }
            },
            None => {
                eprintln!("xtask release: missing version argument\n");
                print_help();
                ExitCode::FAILURE
            }
        },
        "help" | "--help" | "-h" => {
            print_help();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("xtask: unknown subcommand `{other}`\n");
            print_help();
            ExitCode::FAILURE
        }
    }
}

fn print_help() {
    println!(
        "xtask — integral maintainer tooling\n\n\
         USAGE:\n\
         \x20 cargo xtask <SUBCOMMAND>\n\n\
         SUBCOMMANDS:\n\
         \x20 gen-rys-tables   Regenerate crates/integral/src/math/rys_tables.rs from the\n\
         \x20                  in-repo Stieltjes reference (pure Rust, no external library).\n\
         \x20 release <X.Y.Z>  Bump the workspace version (root Cargo.toml) and stamp the\n\
         \x20                  CHANGELOG [Unreleased] section + link footer.\n\
         \x20 help             Show this message.\n"
    );
}
