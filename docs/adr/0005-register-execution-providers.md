# 5. Execution providers, and a speedup that was not real

Status: accepted, with the original claim retracted

## Context

The first end to end question against the real models took 138 seconds to
generate a three sentence answer. The reference plan for this project budgets one
to three seconds, so this was investigated.

Foundry Local runs models through ONNX Runtime, which dispatches work to an
execution provider. Querying the SDK showed none were registered:

    OpenVINOExecutionProvider        is_registered: False
    NvTensorRTRTXExecutionProvider   is_registered: False
    CUDAExecutionProvider            is_registered: False
    WebGpuExecutionProvider          is_registered: False

Registering all four and asking the same question again took 25 seconds. That was
recorded here as a 5.5 times speedup from a setup step.

## The claim was wrong

Two later observations did not fit.

`nvidia-smi` reported 0 percent utilisation and 0 MiB of GPU memory throughout a
full evaluation run. The discrete GPU was never touched. And `discover_eps()` at
the start of every new process reported all four providers unregistered again,
meaning registration does not persist, and every evaluation run had therefore
been executed with no providers registered at all, at 13 to 16 seconds per
answer. Faster than the supposedly accelerated 25 seconds.

The original comparison was not controlled. The 138 second run was the first
generation ever performed on this machine, moments after the model finished
downloading. First run cost and registration were changed at the same time, and
the result was attributed entirely to the second.

`scripts/benchmark_providers.py` separates them. It runs the same question five
times in one process and reports each timing, with and without registering first:

| | first run | median of runs 2 to 5 |
| --- | --- | --- |
| without registering | 30.6 s | **16.6 s** |
| after registering | 14.5 s | **18.7 s** |

Registration does not make generation faster. The gap between the first run and
the rest, visible in both columns, is the effect the original measurement
captured.

## Why registration cannot help here

Every model in the Foundry Local catalog, checked through `model.variants`, ships
exactly one build on this machine, and it is a CPU build:

    qwen3-0.6b             qwen3-0.6b-generic-cpu:4
    qwen3-1.7b             qwen3-1.7b-generic-cpu:2
    qwen3-4b               qwen3-4b-generic-cpu:3
    phi-4-mini             Phi-4-mini-instruct-generic-cpu:5
    qwen3-embedding-0.6b   qwen3-embedding-0.6b-generic-cpu:1

Registering CUDA makes the provider available to the runtime. It does not produce
a CUDA compiled model, and there is no GPU variant to select. `generic-cpu` is
chosen with the providers registered and without them, which the benchmark
confirms by printing the selected variant in both cases.

The RTX 3050 in this machine cannot be used for inference through Foundry Local
as the catalog currently stands. That is a property of the packaged models, not
of the hardware or of a missing setup step.

## Decision

Keep `scripts/check_hardware.py`, but as a diagnostic rather than a required
setup step. Reporting which providers exist, which are registered, and which
variant a model actually selects is what turned a wrong conclusion into a right
one, and it is the first thing to run when generation is slower than expected.

Do not tell users that registration will speed anything up. The README says
generation runs on the CPU and quotes the measured number.

## Consequences

Good:

- The performance numbers now describe what the machine is really doing. A p50 of
  13.6 seconds is CPU inference, and calling it anything else would misrepresent
  every timing in the README.
- The remaining lever for speed is model size, which is measurable and is the
  subject of a separate experiment.

Bad:

- There is no GPU path to pursue on this setup, so the only way to a one to three
  second answer is a smaller model or less context, both of which cost quality.

The lesson worth keeping is procedural rather than technical. The original
measurement changed two variables at once and reported the result as though it
had changed one. It took a contradicting observation, GPU utilisation at zero, to
expose it. A single before and after pair is not a measurement, and this ADR now
records both the wrong conclusion and the controlled test that replaced it.
