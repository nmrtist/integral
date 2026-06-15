//! `release` — bump the workspace version.
//!
//! The version string lives in two places in the root `Cargo.toml`
//! (`[workspace.package]` plus the internal `integral` `[workspace.dependencies]`
//! entry). This subcommand does the whole prep mechanically:
//!
//! 1. rewrite every `version = "<old>"` in the root `Cargo.toml`,
//! 2. rewrite versioned dependency snippets in the public READMEs,
//! 3. refresh `Cargo.lock` via `cargo update --workspace`.
//!
//! Committing, tagging, and publishing stay manual; the next steps are printed.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn run(new_version: &str) -> Result<(), String> {
    validate_version(new_version)?;
    let root = workspace_root();

    let old_version = bump_cargo_toml(&root, new_version)?;
    if old_version == new_version {
        return Err(format!("workspace is already at version {new_version}"));
    }
    bump_readme_versions(&root, &old_version, new_version)?;
    refresh_lockfile(&root);

    println!("\nrelease prep for v{new_version} done (was {old_version}). Next steps:");
    println!("  git commit -am \"chore: prep v{new_version} release\"");
    println!("  git tag v{new_version}");
    println!("  cargo publish -p integral");
    Ok(())
}

/// Accept exactly `MAJOR.MINOR.PATCH` with numeric components.
fn validate_version(v: &str) -> Result<(), String> {
    let parts: Vec<&str> = v.split('.').collect();
    let ok = parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    if ok {
        Ok(())
    } else {
        Err(format!("`{v}` is not a MAJOR.MINOR.PATCH version"))
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits one level under the workspace root")
        .to_path_buf()
}

/// Replace every `version = "<old>"` in the root manifest; returns the old version.
fn bump_cargo_toml(root: &Path, new_version: &str) -> Result<String, String> {
    let path = root.join("Cargo.toml");
    let toml = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;

    let old_version = toml
        .lines()
        .find_map(|l| l.trim().strip_prefix("version = \""))
        .and_then(|rest| rest.strip_suffix('"'))
        .ok_or("no `version = \"...\"` line found in the root Cargo.toml")?
        .to_string();

    let needle = format!("version = \"{old_version}\"");
    let count = toml.matches(&needle).count();
    // 1 in [workspace.package] + the internal `integral` [workspace.dependencies] entry.
    if count != 2 {
        return Err(format!(
            "expected 2 occurrences of `{needle}` in the root Cargo.toml, found {count} — \
             the manifest layout changed; update xtask::release"
        ));
    }

    let bumped = toml.replace(&needle, &format!("version = \"{new_version}\""));
    fs::write(&path, bumped).map_err(|e| format!("write {}: {e}", path.display()))?;
    println!("Cargo.toml: {old_version} -> {new_version} ({count} spots)");
    Ok(old_version)
}

/// Rewrite the versioned dependency snippet in the public READMEs.
fn bump_readme_versions(root: &Path, old_version: &str, new_version: &str) -> Result<(), String> {
    let readmes = [
        root.join("README.md"),
        root.join("crates").join("integral").join("README.md"),
    ];
    let needle = format!("integral = \"{old_version}\"");
    let replacement = format!("integral = \"{new_version}\"");
    for path in readmes {
        let readme =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let count = readme.matches(&needle).count();
        if count != 1 {
            return Err(format!(
                "expected 1 occurrence of `{needle}` in {}, found {count} — update xtask::release",
                path.display()
            ));
        }
        let bumped = readme.replace(&needle, &replacement);
        fs::write(&path, bumped).map_err(|e| format!("write {}: {e}", path.display()))?;
        println!(
            "{}: {old_version} -> {new_version} ({count} spot)",
            path.strip_prefix(root).unwrap_or(&path).display()
        );
    }
    Ok(())
}

/// Refresh the workspace members' versions in Cargo.lock (best-effort).
fn refresh_lockfile(root: &Path) {
    let status = Command::new("cargo")
        .args(["update", "--workspace"])
        .current_dir(root)
        .status();
    match status {
        Ok(s) if s.success() => println!("Cargo.lock: refreshed"),
        _ => eprintln!("warning: `cargo update --workspace` failed; Cargo.lock not refreshed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_validation() {
        assert!(validate_version("0.1.5").is_ok());
        assert!(validate_version("10.20.30").is_ok());
        assert!(validate_version("0.1").is_err());
        assert!(validate_version("0.1.5-rc1").is_err());
        assert!(validate_version("v0.1.5").is_err());
        assert!(validate_version("0..5").is_err());
    }
}
