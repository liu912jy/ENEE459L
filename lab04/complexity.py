from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
)


FLOPS_PER_MAC = 2

# The conventions `to_flops` will honour by name. Anything else is unknown
# rather than an assumption, because the whole point of the parameter is that
# the caller has to say which one they mean.
FLOP_CONVENTIONS = {
    "mac_is_two_flops": 2,
    "mac_is_one_flop": 1,
}

# Batch normalisation holds two learnable vectors per channel (scale and shift)
# and two non-learnable ones (running mean and variance). The first pair are
# parameters; the second pair are buffers. Both are in the file.
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

# Buffers are kept in FP32 even when the weights are not. Halving them saves
# nothing worth having and a denormal running variance is a real failure mode.
BUFFER_DTYPE = "fp32"

# Below this many models there is no line to fit and no residual to report.
MIN_MODELS_FOR_FIT = 3

# Two floats are the same MAC count when they are the same integer. There is no
# tolerance here on purpose: MAC counts are integers, and a tolerance would let
# two genuinely different architectures be reported as tied.
TIE_EXACT = True


# ===========================================================================
# 1. How many numbers are stored
# ===========================================================================

def _layer_parameters(ly: Layer) -> int:
    if ly.kind == "conv":
        c_out = ly.out_shape[0]
        c_in = ly.in_shape[0]

        if ly.kernel is None:
            kh, kw = (1, 1)
        else:
            kh, kw = ly.kernel

        nweights = c_out * (c_in // ly.groups) * kh * kw

        if ly.bias:
            nweights += c_out

        return nweights

    if ly.kind == "linear":
        f_out = ly.out_shape[0]
        f_in = ly.in_shape[0]

        nweights = f_out * f_in

        if ly.bias:
            nweights += f_out

        return nweights

    if ly.kind == "bn":
        return BN_PARAMS_PER_CHANNEL * ly.out_shape[0]

    return 0


def count_parameters(graph: Graph) -> dict[str, Any]:
    per_layer = {}
    total = 0

    for ly in graph:
        n = _layer_parameters(ly)
        per_layer[ly.name] = n
        total += n

    return computed(
        total,
        f"{graph.name}: {len(graph)} layers, shapes from the description",
        per_layer=per_layer,
        includes_bias=True,
        excludes_bn_buffers=True,
        bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
    )


# ===========================================================================
# 2. What those numbers weigh, which is not the size of the file
# ===========================================================================

def model_size_bytes(graph: Graph) -> dict[str, Any]:
  
    """Bytes of stored tensors: parameters plus buffers, at their own dtypes.

    Lecture 04 slide 8 gives the formula as `#Parameters × bit width` and slide
    9 spends a page on why the file on disk is not that number. Three reasons,
    two of which this function has to get right:

      * a model is not stored in one dtype. `Layer.weight_dtype` is per layer
        and a network with FP16 weights and FP32 normalisation is completely
        ordinary. Multiplying a single total by a single bit width is the
        mistake, and on these four descriptions it is worth several per cent
      * buffers are in the file. Batch norm's running statistics are two
        vectors per channel that no optimiser ever touched, and they are still
        bytes you have to ship
      * the container is in the file too — the pickle framing, the state-dict
        keys, the archive directory. This function does *not* try to model
        that, and it says so in `container_overhead_excluded` rather than
        quietly letting the caller assume it did

    Returns a `computed` finding whose value is bytes, with the per-dtype
    breakdown that makes the first bullet checkable.
    """
    per_dtype = {}
    per_layer = {}
    buffer_bytes = 0.0

    for ly in graph:
        n = _layer_parameters(ly)

        param_bytes = n * dtype_bytes(ly.weight_dtype)

        if ly.weight_dtype not in per_dtype:
            per_dtype[ly.weight_dtype] = 0.0

        per_dtype[ly.weight_dtype] += param_bytes
        layer_bytes = param_bytes

        if ly.kind == "bn":
            buffer_elements = BN_BUFFERS_PER_CHANNEL * ly.out_shape[0]
            bn_bytes = buffer_elements * dtype_bytes(BUFFER_DTYPE)

            buffer_bytes += bn_bytes

            if BUFFER_DTYPE not in per_dtype:
                per_dtype[BUFFER_DTYPE] = 0.0

            per_dtype[BUFFER_DTYPE] += bn_bytes
            layer_bytes += bn_bytes

        per_layer[ly.name] = layer_bytes

    total = sum(per_layer.values())

    return computed(
        total,
        f"{graph.name}: per-layer dtypes, buffers at fp32",
        per_layer=per_layer,
        per_dtype=per_dtype,
        buffer_bytes=buffer_bytes,
        container_overhead_excluded=True,
        note="not the size of the file on disk; see the handout, Stage A step 3",
    )


# ===========================================================================
# 3. The memory nobody puts in the table
# ===========================================================================

def _elements(shape: tuple[int, ...]) -> int:
    total = 1

    for d in shape:
        total *= d

    return total


def _last_use(graph: Graph) -> dict[str, int]:
    last = {}
    names = []

    for ly in graph.layers:
        names.append(ly.name)

    for i, ly in enumerate(graph.layers):
        if ly.reads:
            for name in ly.reads:
                last[name] = i
        else:
            if i == 0:
                last["__input__"] = 0
            else:
                last[names[i - 1]] = i

        if ly.name not in last:
            last[ly.name] = i

    if len(graph.layers) > 0:
        last[graph.layers[-1].name] = len(graph) - 1

    return last


def _peak_elements(graph: Graph, last_use: dict[str, int]) -> int:
    live = {}
    live["__input__"] = _elements(graph.input_shape)

    peak = sum(live.values())

    for i, ly in enumerate(graph.layers):
        live[ly.name] = ly.out_elements

        current = sum(live.values())

        if current > peak:
            peak = current

        for name in list(live.keys()):
            if last_use.get(name) == i:
                del live[name]

    return peak
  
def count_activations(graph: Graph) -> dict[str, Any]:
    """Total and peak activation footprint, in elements and in bytes.

    UNC COMP 790-150 Lec 2 p. 70 gives AlexNet as total 932,264 and peak
    440,928, and the two numbers answer two different questions. Total is what
    the whole forward pass produced. Peak is how much had to be resident at
    once, and peak is the one that decides whether the model runs.

    Peak is not `max(out_elements)`. Three things make it larger than that:

      * a layer's input is still resident while its output is being written.
        The live set at layer *i* contains both
      * a tensor consumed by a later layer stays resident in between. `add`
        layers name two inputs in `Layer.reads`, and the earlier one has been
        sitting in memory across every layer of the block. This is the residual
        connection and it is the single largest contributor to peak in
        ResNet-shaped networks
      * the network's own input is a tensor too

    The implementation is a liveness pass: work out the last layer that reads
    each tensor, then walk forward keeping a live set and taking the maximum of
    its total size. Anything simpler than that is wrong on any graph with a
    skip connection, and it is wrong quietly, in the direction that says the
    model fits.

    Returns a `computed` finding whose value is peak *bytes*, because bytes are
    what a memory budget is denominated in, with elements and the layer where
    the peak occurs alongside.
    """
    last_use = _last_use(graph)

    live = {}
    live["__input__"] = (
        _elements(graph.input_shape) * dtype_bytes(graph.precision)
    )

    peak_bytes = sum(live.values())
    peak_at = "__input__"

    total_elements = 0
    total_bytes = 0.0

    for i, ly in enumerate(graph.layers):
        out_elements = ly.out_elements
        out_b = out_elements * dtype_bytes(ly.act_dtype)

        live[ly.name] = out_b

        total_elements += out_elements
        total_bytes += out_b

        resident = sum(live.values())

        if resident > peak_bytes:
            peak_bytes = resident
            peak_at = ly.name

        for name in list(live.keys()):
            if last_use.get(name) == i:
                del live[name]

    peak_elements = _peak_elements(graph, last_use)

    return computed(
        peak_bytes,
        f"{graph.name}: liveness over {len(graph)} layers, input included",
        peak_at=peak_at,
        peak_elements=peak_elements,
        total_elements=total_elements,
        total_bytes=total_bytes,
        includes_network_input=True,
        note="peak is the resident set, not the largest single tensor",
    )


# ===========================================================================
# 4. The factor of two that halves everybody's numbers
# ===========================================================================

def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops") -> dict[str, Any]:
    """Convert a MAC finding to a FLOP finding, naming the convention used.

    A multiply-accumulate is one multiply and one add, so it is two
    floating-point operations. Roughly half the published literature calls a
    MAC one FLOP anyway, and the two conventions differ by exactly the factor
    that makes two papers' numbers incomparable.

    Three requirements, and the third is the graded one:

      * multiply once. `FLOPS_PER_MAC` exists so that the number 2 appears in
        this file exactly once
      * an unknown MAC count converts to an unknown FLOP count. It does not
        convert to zero and it does not raise
      * the convention goes in the finding. A FLOP count that does not say
        which convention produced it is not a FLOP count, it is a number, and
        `to_flops(x, "mac_is_one_flop")` has to be as clearly labelled as the
        default

    An unrecognised convention is `unknown`, not a default. The caller asked
    for something this function does not know how to do.
    """
    pass
  
def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops")->dict[str, Any]:
    """Convert a MAC finding to a FLOP finding, naming the convention used.

    A multiply-accumulate is one multiply and one add, so it is two
    floating-point operations. Roughly half the published literature calls a
    MAC one FLOP anyway, and the two conventions differ by exactly the factor
    that makes two papers' numbers incomparable.

    Three requirements, and the third is the graded one:

      * multiply once. `FLOPS_PER_MAC` exists so that the number 2 appears in
        this file exactly once
      * an unknown MAC count converts to an unknown FLOP count. It does not
        convert to zero and it does not raise
      * the convention goes in the finding. A FLOP count that does not say
        which convention produced it is not a FLOP count, it is a number, and
        `to_flops(x, "mac_is_one_flop")` has to be as clearly labelled as the
        default

    An unrecognised convention is `unknown`, not a default. The caller asked
    for something this function does not know how to do.
    """

    if not is_answered(macs):
        return unknown(
            "MAC count",
            "no valid MAC count was provided"
        )

    if convention not in FLOP_CONVENTIONS:
        return unknown(
            "FLOP convention",
            f"unrecognized convention {convention}"
        )

    factor = FLOP_CONVENTIONS[convention]
    total = macs["value"] * factor

    per_layer = {}

    if "per_layer" in macs:
        for name in macs["per_layer"]:
            per_layer[name] = macs["per_layer"][name] * factor

    return computed(
        total,
        macs["source"],
        convention=convention,
        flops_per_mac=factor,
        per_layer=per_layer,
        note="a count of operations contains no unit of time",
    )