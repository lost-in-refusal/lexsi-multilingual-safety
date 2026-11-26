### The base model in four loadings: memory, speed and the benchmarks beside them (section 22)

Tiny Aya Global with no adapter. bfloat16; NF4 (bitsandbytes 4-bit, double quantization, compute in bfloat16, the loader of section 12.2); uniform 4-bit and the circuit plan (3 layers at 8 bits, 27 at 4 and 6 at 2, the tier size of section 15) through CircuitKIT with optimum-quanto, weights only, applied on the CPU. Every figure is marked with the run it comes from:

- (a) section 15: Apple laptop GPU (MPS); quanto weights written out as plain bfloat16 numbers.
- (b) sections 12.0 and 12.2: NVIDIA L4; bfloat16 from job control (arm gpu-base), NF4 from job nf4 (arm gpu-base-nf4).
- (c) section 15b: NVIDIA L4; quanto weights written out as plain bfloat16 numbers.
- (d) this job: NVIDIA A100-SXM4-80GB; quanto weights packed on the GPU, upload `20261126-061511-be37c5c`, commit `be37c5c`.

| | bfloat16 | NF4 (bitsandbytes) | Uniform 4-bit (quanto) | Circuit plan (quanto) |
|---|---:|---:|---:|---:|
| Memory after loading (MiB) | 6,496 (d) | 2,418 (d) | 2,475 (d) | 2,470 (d) |
| Peak memory during perplexity (MiB) | 7,437 (d) | 3,358 (d) | 3,416 (d) | 3,410 (d) |
| Peak memory at batch 8 (MiB) | 7,045 (d) | 2,966 (d) | 3,025 (d) | 3,021 (d) |
| Output tokens per second, batch 1 (tokens/s) | 24.4 (d) | 17.3 (d) | 20.4 (d) | 16.6 (d) |
| Median tokens per second of a prompt, batch 1 (tokens/s) | 24.5 (d) | 17.8 (d) | 20.6 (d) | 16.5 (d) |
| Output tokens per second, batch 8 (tokens/s) | 142.1 (d) | 100.4 (d) | 114.1 (d) | 87.9 (d) |
| Perplexity of the 300 task references | 6.22 (d) | 6.39 (d) | 6.49 (d) | 7.81 (d) |
| Strict attack success, English (%) | 7.2 (a) / 5.5 (b) | 11.0 (b) | 6.2 (a) | 8.9 (a) |
| Strict attack success, Korean (%) | 5.2 (a) / 5.5 (b) | 7.9 (b) | 7.9 (a) | 11.7 (a) |
| Strict attack success, Chinese (%) | 9.3 (a) / 7.9 (b) | 9.6 (b) | 8.2 (a) | 13.4 (a) |
| Strict attack success, Swahili (%) | 5.5 (a) / 5.2 (b) | 7.2 (b) | 5.5 (a) | 5.8 (a) |
| Strict attack success, mean of the four languages (%) | 6.8 (a) / 6.0 (b) | 8.9 (b) | 7.0 (a) | 10.0 (a) |
| Refusals of XSTest's safe prompts, English (%) | 9.6 (a) / 9.6 (b) | 7.6 (b) | 8.4 (a) | 18.8 (a) |
| Refusals of XSTest's safe prompts, Korean (%) | 7.6 (a) / 8.4 (b) | 8.8 (b) | 10.0 (a) | 28.4 (a) |
| Refusals of XSTest's safe prompts, Chinese (%) | 8.8 (a) / 8.4 (b) | 6.4 (b) | 8.4 (a) | 20.8 (a) |
| Refusals of XSTest's safe prompts, Swahili (%) | 7.6 (a) / 8.4 (b) | 5.6 (b) | 8.4 (a) | 10.0 (a) |
| Task score (1 to 7, 300 task prompts) | 6.35 (a) / 6.29 (d) | 6.20 (d) | 6.09 (a) | 4.78 (a) |
| Global-MMLU-Lite accuracy, four languages (%) | 54.5 (c) / 54.8 (d) | 51.4 (d) | 51.2 (c) | 37.1 (c) |

Memory is torch.cuda.memory_allocated() after loading and the peak of max_memory_allocated() during a phase; the CUDA context and the allocator's cache are not in it. Speed: greedy generation of at most 256 new tokens for the first 50 task prompts in file order. At batch 1 one prompt at a time; at batch 8 the same prompts sorted longest first, as lir.generate sorts them, with left padding. Before each, one warm-up call on prompts outside the 50 is made and not counted. Output tokens per second is the new tokens, counted as lir.generate counts them, over the seconds of the generate calls with the GPU synchronized. The answers differ in length between arms, so the figures rest on different numbers of tokens: bfloat16 11,506 at batch 1; NF4 (bitsandbytes) 10,773 at batch 1; Uniform 4-bit (quanto) 11,148 at batch 1; Circuit plan (quanto) 10,821 at batch 1.

The strict attack success over the four languages is not stored in quantization.json or gpu.json: it is the plain mean of the four languages' rates, which have the same 291 prompt families each; for section 15 uniform 4-bit minus bfloat16 of these means is Q0's mean.

On this job's GPU the task score is the mean over the answers the judge scored: bfloat16 300 of 300, NF4 (bitsandbytes) 300 of 300. Section 15 read an arm's task score only when all 300 were scored.

The quanto weights on the GPU, by tier (class of the weight tensor and number of layers): Uniform 4-bit (quanto): mid TinyGemmWeightQBitsTensor 252; Circuit plan (quanto): mid TinyGemmWeightQBitsTensor 189, high WeightQBytesTensor 21, low WeightQBitsTensor 42; 2-bit unpacking by quanto_cuda kernel.

Not measured on this job's GPU: the safety sets, and the task score and Global-MMLU-Lite of uniform 4-bit and the circuit plan. The table holds the base model only.
