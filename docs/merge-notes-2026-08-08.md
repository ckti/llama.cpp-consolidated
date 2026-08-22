# Consolidated Upstream Merge Notes

Date: 2026-08-08

## Source revisions

- `ggml-org/llama.cpp` `master` at `69bf6437914596fbbc4caf09a7ac16f2acdd1a94`
- `TheTom/llama-cpp-turboquant` `feature/turboquant-kv-cache` at `2f2f32f5d9517518c9e860f30131acb09840a965`
- `PrismML-Eng/llama.cpp` `prism` at `9ca265a57f85f2117942490f421f64a226dd9847`
- `Mintplex-Labs/prism-ml-llama.cpp` `prism` at `520d93d8a8fd0ac84c0fa92d4568a68b14d495f0`

Merge commits:

- `348b7634d` merged the current ggml.org upstream.
- `6559decee` merged the current TurboQuant feature branch.

The ML-Eng and Mintplex revisions were already ancestors of the target branch, so Git reported both as already up to date and created no duplicate merge commits.

## File changes

The ggml.org merge changed 303 paths. The changes are grouped below by subsystem.

| Area | Files or directories | Change summary |
| --- | --- | --- |
| CI and release | `.github/workflows/`, `ci/run.sh`, `build-xcframework.sh`, `SECURITY.md` | Updated build, release, platform, and security workflow configuration. |
| Common runtime | `common/arg.*`, `common/common.*`, `common/download.*`, `common/sampling.*`, `common/fit.cpp`, `common/jinja/` | Added and revised command-line options, download handling, sampling behavior, prompt/template handling, and fitting support. |
| Conversion | `conversion/`, especially `conversion/qwen.py`, `conversion/deepseek.py`, `conversion/glm.py`, and `conversion/qwen3tts.py` | Added model conversion support and metadata handling for newer DeepSeek, Qwen, GLM, and Qwen3 TTS models. |
| Documentation and operations | `docs/backend/SYCL.md`, `docs/ops*`, `examples/sycl/`, model-conversion scripts | Refreshed backend operation tables, SYCL documentation, and example scripts. |
| GGML backends | `ggml/src/ggml-cpu/`, `ggml/src/ggml-cuda/`, `ggml/src/ggml-metal/`, `ggml/src/ggml-opencl/`, `ggml/src/ggml-sycl/`, `ggml/src/ggml-vulkan/` | Added backend operations and kernels, updated device handling, improved Vulkan submission behavior, and added newer SYCL/Vulkan functionality including DSV4/GLA support. |
| GGUF and public API | `gguf-py/`, `include/llama.h`, `src/llama-ext.h` | Updated GGUF types, reader/writer validation, tensor mapping, and public model/context APIs. |
| Core model runtime | `src/llama-*.cpp`, `src/llama-*.h`, `src/models/` | Added or updated model architectures, graph construction, KV-cache variants, sampler behavior, model loading, and DFlash/DSpark-related integration. |
| Templates | `models/templates/` | Added the DeepSeek V4 Flash template and updated the DeepSeek V4 template. |
| Tests | `tests/` | Expanded argument, backend, sampler, chat, grammar, model-resolution, and API coverage. |
| Multimodal and server tools | `tools/mtmd/`, `tools/server/`, `tools/tts/` | Added Qwen3 TTS and multimodal support, server API/tool changes, router/proxy coverage, and TTS updates. |
| Web UI | `tools/ui/` | Updated the contenteditable chat form, working-directory and picker workflows, tool-call rendering, utilities, tests, and package metadata. |
| Vendor | `vendor/cpp-httplib/`, `vendor/sheredom/`, `scripts/sync_vendor.py` | Updated vendored dependencies and added portable subprocess patches while retaining the signal-status patch used by consolidated. |

## TurboQuant refresh

The TurboQuant merge added the UI asset completeness validation in `scripts/ui-assets.cmake` and synchronized the related `src/llama-context.cpp` change. The target already contained the DFlash allocation and typed-copy fixes from the feature branch, so those conflict sections were retained from consolidated rather than replacing the newer DSpark/DFlash implementation with the older branch version.

## Conflict resolution policy

Where the branches overlapped, consolidated’s newer DSpark, DFlash, speculative-decoding, TurboQuant, and public API implementations were preserved. Upstream Vulkan scheduling and token-embedding API additions were incorporated. The resulting target is clean, and all four source revisions are ancestors of `HEAD`.
