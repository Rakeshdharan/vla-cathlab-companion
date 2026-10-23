"""Array wire format for the policy server (stdlib + numpy): base64 of the C-contiguous bytes + shape + dtype."""
import base64
import numpy as np

DTYPES = ("uint8", "float32", "float64", "int64")


def encode_array(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    if a.dtype.name not in DTYPES:
        raise ValueError(f"unsupported dtype {a.dtype.name}; use one of {DTYPES}")
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape), "dtype": a.dtype.name}


def decode_array(d: dict) -> np.ndarray:
    if d.get("dtype") not in DTYPES:
        raise ValueError(f"unsupported dtype {d.get('dtype')!r}")
    flat = np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"])
    shape = [int(x) for x in d["shape"]]
    if int(np.prod(shape)) != flat.size:
        raise ValueError(f"shape {shape} does not match {flat.size} values")
    return flat.reshape(shape).copy()
