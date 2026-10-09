# Consolidated Upstream Merge Notes

Date: 2026-08-21

## Merge commits

| Commit | Parent used for the file comparison | Source tip | Paths changed |
| --- | --- | --- | ---: |
| `fe48cd3e4` | `b0c6f1941` | upstream `master` at `9a286ac98` | 1,184 |
| `79d10ab5d` | `fe48cd3e4` | TurboQuant at `e30664a71` | 91 |

The ML-Eng `prism` tip `9ca265a57` and Mintplex `prism` tip `520d93d8a` were already ancestors of the target, so Git reported both as already up to date and created no additional merge commits for them.

## Upstream merge: `fe48cd3e4`

This merge brought `ggml-org/llama.cpp` master through `9a286ac98` into the existing consolidated tree. Consolidated-side content was retained for conflicting paths.

| Area | Paths changed | Summary |
| --- | ---: | --- |
| Web UI and server tools | 707 | Updated UI structure, chat workflows, settings, stores, utilities, tests, server behavior, and tool documentation. |
| GGML backends | 178 | Updated CPU, CUDA, Metal, SYCL, Vulkan, and shared backend code and kernels. |
| Model conversion | 86 | Added and updated conversion support and model metadata handling. |
| Core runtime and models | 52 | Updated graph, context, architecture, model loading, sampling, KV-cache, and model implementations. |
| Common runtime | 29 | Updated argument parsing, chat, downloads, sampling, templates, and shared utilities. |
| Vendor and build support | 22 vendor, 22 examples, 21 CI, 15 scripts, 4 CMake | Updated dependencies, examples, workflows, synchronization scripts, and build configuration. |
| Tests and documentation | 16 tests, 11 docs, 4 models, 4 GGUF-Py | Added or updated tests, templates, operational documentation, and GGUF tooling. |
| Other project files | Remaining paths | Updated public headers, packaging, requirements, metadata, and project documentation. |

## TurboQuant merge: `79d10ab5d`

This merge brought TurboQuant tip `e30664a71` into the upstream-merged tree.

| Area | Paths changed | Summary |
| --- | ---: | --- |
| CUDA and GGML backends | 43 | Added MOE-cache support and updated CUDA, Metal, Vulkan, CPU, and shared backend paths. |
| Core runtime and models | 11 | Updated context, graph, KV-cache, model, and Qwen-related runtime code. |
| Tools and tests | 11 tools, 8 tests | Added MOE-cache, speculative-adaptive, benchmark, server, and integration coverage. |
| Common and documentation | 7 common, 2 docs | Added adaptive speculative support and refreshed related documentation. |
| Build and metadata | Remaining paths | Updated public headers, GGUF metadata, scripts, skills, vendor configuration, and project instructions. |

## Exact file manifests

The complete file lists, including status codes and rename paths, are generated directly from the merge commits with:

```bash
git diff --name-status b0c6f1941 fe48cd3e4
git diff --name-status fe48cd3e4 79d10ab5d
```

These comparisons use each merge commit's first parent, so they document the files introduced or changed by each integration in `consolidated`.
