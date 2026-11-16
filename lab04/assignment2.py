from dataclasses import dataclass
from typing import Any


# ============================================================
# COPIED FROM LAB04 graph.py
# ============================================================

DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}


@dataclass
class Layer:
    name: str
    kind: str
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    kernel: tuple[int, int] | None = None
    stride: int = 1
    groups: int = 1
    bias: bool = False
    weight_dtype: str = "fp32"
    act_dtype: str = "fp32"
    reads: tuple[str, ...] = ()

    @property
    def in_elements(self) -> int:
        n = 1
        for x in self.in_shape:
            n *= x
        return n

    @property
    def out_elements(self) -> int:
        n = 1
        for x in self.out_shape:
            n *= x
        return n


@dataclass
class Graph:
    name: str
    input_shape: tuple[int, ...]
    layers: list[Layer]
    precision: str = "fp32"

    def __iter__(self):
        return iter(self.layers)

    def get(self, name: str) -> Layer | None:
        for ly in self.layers:
            if ly.name == name:
                return ly
        return None

def dtype_bytes(dtype: str) -> float:
    return DTYPE_BYTES[dtype]


def computed(value: Any, source: str, **extra: Any) -> dict[str, Any]:
    return {"value": value, "source": source, "status": "computed", **extra}


# ============================================================
# COPIED FROM LAB04 complexity.py
# ============================================================

FLOPS_PER_MAC = 2
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2
BUFFER_DTYPE = "fp32"


def _layer_parameters(ly: Layer) -> int:
    if ly.kind == "conv":
        c_out = ly.out_shape[0]
        c_in = ly.in_shape[0]
        kh, kw = ly.kernel if ly.kernel is not None else (1, 1)
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



# ============================================================
# New Functions for Assignment 2
# ============================================================

def _layer_macs(ly: Layer) -> int:
    if ly.kind == "conv":
        kh, kw = (1, 1)
        if ly.kernel is not None:
            kh, kw = ly.kernel

        return ly.out_elements * (ly.in_shape[0] // ly.groups) * kh * kw

    if ly.kind == "linear":
        return ly.in_elements * ly.out_elements

    return 0


def arithmetic_intensity(graph: Graph) -> dict[str, Any]:
    per_layer = {}
    total_flops = 0
    total_bytes = 0.0
    layers = list(graph)

    for i, ly in enumerate(layers):

        if ly.reads:
            in_dtype = graph.get(ly.reads[0]).act_dtype
        elif i > 0:
            in_dtype = layers[i - 1].act_dtype
        else:
            in_dtype = graph.precision

        input_bytes = ly.in_elements * dtype_bytes(in_dtype)

        if ly.kind == "add" and len(ly.reads) == 2:
            input_bytes *= 2

        weight_bytes = _layer_parameters(ly) * dtype_bytes(ly.weight_dtype)

        if ly.kind == "bn":
            buffer_bytes = BN_BUFFERS_PER_CHANNEL * ly.out_shape[0] * dtype_bytes(BUFFER_DTYPE)
            weight_bytes += buffer_bytes

        output_bytes = ly.out_elements * dtype_bytes(ly.act_dtype)

        layer_bytes = input_bytes + weight_bytes + output_bytes
        layer_flops = _layer_macs(ly) * FLOPS_PER_MAC

        intensity = 0.0
        if layer_bytes > 0:
            intensity = round(layer_flops / layer_bytes, 4)

        per_layer[ly.name] = {
            "flops": layer_flops,
            "bytes": float(layer_bytes),
            "intensity": intensity
        }

        total_flops += layer_flops
        total_bytes += layer_bytes

    value = 0.0
    if total_bytes > 0:
        value = round(total_flops / total_bytes, 4)

    return computed(
        value,
        f"{graph.name}: FLOPs over bytes, every byte counted once",
        per_layer=per_layer,
        total_flops=total_flops,
        total_bytes=float(total_bytes),
        flops_per_mac=FLOPS_PER_MAC,
        cache_reuse_assumed=False,
        note="no fusion and no cache reuse; both would lower the byte column"
    )


# ============================================================
# COPIED FROM LAB04 main.py
# ============================================================

def build_synthetic_graph() -> Graph:
    layers = [
        Layer("conv1", "conv", (3, 32, 32), (16, 32, 32), kernel=(3, 3)),
        Layer("bn1", "bn", (16, 32, 32), (16, 32, 32)),
        Layer("relu1", "relu", (16, 32, 32), (16, 32, 32)),
        Layer("res_conv", "conv", (16, 32, 32), (16, 32, 32), kernel=(3, 3), reads=("relu1",)),
        Layer("add1", "add", (16, 32, 32), (16, 32, 32), reads=("relu1", "res_conv")),
        Layer("pool", "pool", (16, 32, 32), (16, 1, 1)),
        Layer("flatten", "flatten", (16, 1, 1), (16,)),
        Layer("fc", "linear", (16,), (10,), bias=True),
    ]
    return Graph("toy_resnet", (3, 32, 32), layers, "fp32")


if __name__ == "__main__":
    import json
    graph = build_synthetic_graph()
    result = arithmetic_intensity(graph)
    print(json.dumps(result, indent=2))
