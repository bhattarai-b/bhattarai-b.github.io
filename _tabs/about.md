---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

> ML inference systems engineer focused on performance, profiling, and hardware-aware optimization. I work on making real workloads faster and more predictable through careful measurement and targeted systems changes.

I work on machine learning inference performance and systems optimization, primarily focused on understanding why workloads behave the way they do on real hardware and improving them through measurement-driven changes.

My background sits at the intersection of models, runtimes, and hardware. I spend most of my time profiling inference pipelines, reasoning about execution at the CPU / accelerator level, and tracing performance issues across memory hierarchy, scheduling, batching, and runtime configuration. I’m comfortable working bottom-up from low-level metrics as well as top-down from service-level latency and throughput goals.

I’ve worked on optimizing and benchmarking inference workloads using CPUs and accelerators, with hands-on experience debugging issues related to cache behavior, NUMA placement, memory allocation, kernel execution, and containerized environments. When performance is off, I focus on isolating the bottleneck, validating hypotheses with data, and applying targeted fixes rather than broad rewrites.

The kinds of problems I’m most interested in include:

Inference latency and throughput tradeoffs under production constraints

Hardware-aware optimization (memory locality, parallelism, scheduling, batching)

Runtime and serving behavior (model formats, execution graphs, resource utilization)

Building repeatable evaluation and profiling setups for inference workloads

This repository is a collection of benchmarks, experiments, and notes that reflect how I work: reduce problems to something measurable, understand the system end-to-end, and document what actually moves the needle. It’s intentionally practical and occasionally unfinished — the focus is learning through real workloads rather than polished demos.

I’m particularly motivated by roles where performance, efficiency, and reliability directly impact product viability — whether that’s optimizing kernels and runtimes at a hardware vendor, or improving inference platforms that serve models at scale.