"""Fast GF(2^8) arithmetic on numpy arrays.

The field uses the irreducible polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D),
which is the same default field as ``galois.GF(2**8)``, so results match
the earlier galois-based scripts byte for byte.

Addition and subtraction are XOR. Multiplication by a scalar is a single
table lookup over the whole array, which is what makes image-sized
sharing fast: every operation in Shamir / Reed-Solomon decoding with
fixed x-coordinates is "scalar times array".
"""
from __future__ import annotations

import numpy as np

PRIM_POLY = 0x11D
GENERATOR = 2

EXP = np.zeros(512, dtype=np.uint8)
LOG = np.zeros(256, dtype=np.int32)

_v = 1
for _i in range(255):
    EXP[_i] = _v
    LOG[_v] = _i
    _v <<= 1
    if _v & 0x100:
        _v ^= PRIM_POLY
EXP[255:510] = EXP[0:255]

# Full 256 x 256 multiplication table: MUL[a, b] = a * b in GF(2^8).
_a = np.arange(256)
MUL = EXP[(LOG[_a][:, None] + LOG[_a][None, :]) % 255].astype(np.uint8)
MUL[0, :] = 0
MUL[:, 0] = 0

INV = np.zeros(256, dtype=np.uint8)
INV[1:] = EXP[(255 - LOG[np.arange(1, 256)]) % 255]


def mul(a: int, b: int) -> int:
    return int(MUL[a, b])


def inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in GF(2^8)")
    return int(INV[a])


def div(a: int, b: int) -> int:
    return mul(a, inv(b))


def power(a: int, e: int) -> int:
    r = 1
    for _ in range(e):
        r = mul(r, a)
    return r


def scale(c: int, arr: np.ndarray) -> np.ndarray:
    """c * arr for a scalar field element c and a uint8 array."""
    return MUL[c][arr]


def lagrange_weights(xs: list[int], at: int) -> list[int]:
    """Weights w_j such that f(at) = sum_j w_j * f(xs[j]) for deg f < len(xs)."""
    ws = []
    for j, xj in enumerate(xs):
        num, den = 1, 1
        for m, xm in enumerate(xs):
            if m != j:
                num = mul(num, at ^ xm)
                den = mul(den, xj ^ xm)
        ws.append(div(num, den))
    return ws


def invert_matrix(m: list[list[int]]) -> list[list[int]]:
    """Gauss-Jordan inverse of a small square matrix over GF(2^8)."""
    n = len(m)
    a = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r][col] != 0)
        a[col], a[piv] = a[piv], a[col]
        pinv = inv(a[col][col])
        a[col] = [mul(pinv, v) for v in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [v ^ mul(f, w) for v, w in zip(a[r], a[col])]
    return [row[n:] for row in a]


def vandermonde_inverse(xs: list[int]) -> list[list[int]]:
    """Inverse of V where V[j][i] = xs[j]^i; maps share values -> coefficients."""
    k = len(xs)
    v = [[power(x, i) for i in range(k)] for x in xs]
    return invert_matrix(v)


def combine(weights: list[int], arrays: list[np.ndarray]) -> np.ndarray:
    """sum_j weights[j] * arrays[j] over GF(2^8)."""
    out = np.zeros_like(arrays[0])
    for w, arr in zip(weights, arrays):
        if w:
            out ^= MUL[w][arr]
    return out
