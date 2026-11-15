# Publishing integral

1. Run `cargo xtask release X.Y.Z`, review the version changes, and get the
   release commit onto `main`.
2. Check out the release commit locally with a clean working tree and run
   `cargo test --workspace --release --locked`.
3. Once the tests pass, run `git tag vX.Y.Z` and `git push origin vX.Y.Z`.
   GitHub Actions checks the tagged commit and version, runs the tests, and
   publishes `integral` to crates.io.
4. Confirm the version on crates.io.
