# TGV Denoising in Rust

This is a Rust implementation of the TGV denoising algorithm.

Meant as a learning exercise to understand scientific computing in Rust.

Basic benchmark (M1 Mac Mini Late 2020):
Rust: 4.34 seconds
Python: 4.80 seconds

Parallelized version working on 32x32 patches:
Rust (rayon): 0.37 seconds
Python (joblib): 2.2 seconds

Roadmap:
- [x] Basic working version
- [x] Add basic benchmarks vs Python implementation
- [X] Parallelize the denoising on patches
- [X] Parallel benchmarks vs Python implementation
- [x] Package as Python module
- [ ] Handle patch overlaps to avoid edge effects
- [ ] Package as CLI tool

