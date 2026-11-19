"""ROI-adaptive ramp secret sharing for images, with Reed-Solomon cheater
identification and proactive share refresh. All arithmetic is in GF(2^8).

Scheme summary
--------------
* The image bytes (H x W x C, raster order) are split into two regions:
  ``roi`` (the question area, a rectangle) and ``bg`` (everything else).
* Each region is cut into groups of L consecutive bytes. For every group a
  polynomial of degree k-1 is built:
      f(x) = a_0 + a_1 x + ... + a_{k-1} x^{k-1}
  where a_0..a_{L-1} are the L secret bytes and a_L..a_{k-1} are random.
  L = 1 is Shamir (perfect secrecy, share = image size),
  L = k is Thien-Lin (share = 1/k of image, but leaks structure),
  1 < L < k is a ramp scheme (any k-L shares reveal nothing).
* Share x (x = 1..n) stores f(x) for every group: one byte per group.
* f(1..n) is a Reed-Solomon codeword of length n and dimension k, so with
  m > k shares the decoder detects up to m-k bad shares and corrects up to
  floor((m-k)/2) of them, per group, which also gives a tamper map.
"""
from __future__ import annotations

import io
import json
import secrets
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np
from PIL import Image, PngImagePlugin

from . import gf256 as gf

MAX_SHARES = 8
REGIONS = ("bg", "roi")


# --------------------------------------------------------------------------
# Parameters and layout
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    n: int
    k: int
    L_roi: int
    L_bg: int
    shape: tuple[int, int, int]                 # H, W, C
    roi: tuple[int, int, int, int] | None       # x0, y0, x1, y1 (pixels, exclusive end)

    def L(self, region: str) -> int:
        return self.L_roi if region == "roi" else self.L_bg

    def validate(self) -> None:
        if not 2 <= self.k <= self.n <= MAX_SHARES:
            raise ValueError(f"need 2 <= k <= n <= {MAX_SHARES}")
        for L in (self.L_roi, self.L_bg):
            if not 1 <= L <= self.k:
                raise ValueError("L must be between 1 and k")


def random_bytes(count: int) -> np.ndarray:
    """Cryptographically secure random bytes (not np.random)."""
    return np.frombuffer(secrets.token_bytes(count), dtype=np.uint8).copy()


def roi_pixel_mask(params: Params) -> np.ndarray:
    H, W, _ = params.shape
    mask = np.zeros((H, W), dtype=bool)
    if params.roi is not None:
        x0, y0, x1, y1 = params.roi
        mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    return mask


def byte_mask_from_pixels(pix_mask: np.ndarray, channels: int) -> np.ndarray:
    return np.repeat(pix_mask[:, :, None], channels, axis=2).ravel()


def layout(params: Params) -> dict[str, np.ndarray]:
    """For each region, an int array (G, L) of flat byte positions; -1 = padding."""
    C = params.shape[2]
    roi_bytes = byte_mask_from_pixels(roi_pixel_mask(params), C)
    out = {}
    for region in REGIONS:
        pos = np.flatnonzero(roi_bytes if region == "roi" else ~roi_bytes)
        L = params.L(region)
        pad = (-len(pos)) % L
        pos = np.concatenate([pos, np.full(pad, -1, dtype=pos.dtype)])
        out[region] = pos.reshape(-1, L)
    return out


# --------------------------------------------------------------------------
# Shares
# --------------------------------------------------------------------------
@dataclass
class Share:
    x: int
    params: Params
    paper_id: str
    epoch: int
    custodian: str
    streams: dict[str, np.ndarray] = field(repr=False)

    @property
    def nbytes(self) -> int:
        return int(sum(s.size for s in self.streams.values()))

    def copy(self) -> "Share":
        return replace(self, streams={r: s.copy() for r, s in self.streams.items()})

    # ---- file format: grayscale PNG + JSON metadata in a tEXt chunk -----
    def to_png_bytes(self) -> bytes:
        flat = np.concatenate([self.streams[r] for r in REGIONS])
        width = max(1, int(np.ceil(np.sqrt(flat.size))))
        height = int(np.ceil(flat.size / width))
        canvas = np.zeros(width * height, dtype=np.uint8)
        canvas[: flat.size] = flat
        meta = {
            "scheme": "roi-ramp-sss/1", "x": self.x, "paper_id": self.paper_id,
            "epoch": self.epoch, "custodian": self.custodian,
            "n": self.params.n, "k": self.params.k,
            "L_roi": self.params.L_roi, "L_bg": self.params.L_bg,
            "shape": list(self.params.shape),
            "roi": list(self.params.roi) if self.params.roi else None,
            "lengths": {r: int(self.streams[r].size) for r in REGIONS},
        }
        info = PngImagePlugin.PngInfo()
        info.add_text("sss-exam", json.dumps(meta))
        buf = io.BytesIO()
        Image.fromarray(canvas.reshape(height, width)).save(buf, "PNG", pnginfo=info)
        return buf.getvalue()

    @staticmethod
    def from_png_bytes(data: bytes) -> "Share":
        img = Image.open(io.BytesIO(data))
        if "sss-exam" not in img.info:
            raise ValueError("not a share file (missing sss-exam metadata)")
        meta = json.loads(img.info["sss-exam"])
        flat = np.asarray(img.convert("L"), dtype=np.uint8).ravel()
        params = Params(n=meta["n"], k=meta["k"], L_roi=meta["L_roi"], L_bg=meta["L_bg"],
                        shape=tuple(meta["shape"]), roi=tuple(meta["roi"]) if meta["roi"] else None)
        streams, off = {}, 0
        for r in REGIONS:
            ln = meta["lengths"][r]
            streams[r] = flat[off: off + ln].copy()
            off += ln
        return Share(x=meta["x"], params=params, paper_id=meta["paper_id"], epoch=meta["epoch"],
                     custodian=meta["custodian"], streams=streams)


def eval_poly(coeffs: list[np.ndarray], x: int) -> np.ndarray:
    return gf.combine([gf.power(x, i) for i in range(len(coeffs))], coeffs)


def split(image: np.ndarray, params: Params, custodians: list[str] | None = None) -> list[Share]:
    """Create n shares of an H x W x C uint8 image."""
    params.validate()
    if image.shape != params.shape:
        raise ValueError(f"image shape {image.shape} != params.shape {params.shape}")
    flat = np.concatenate([image.ravel(), np.zeros(1, dtype=np.uint8)])  # index -1 -> 0 pad
    paper_id = secrets.token_hex(6)
    streams_per_x = {x: {} for x in range(1, params.n + 1)}
    for region, pos in layout(params).items():
        G, L = pos.shape
        coeffs = [flat[pos[:, i]] for i in range(L)]
        coeffs += [random_bytes(G) for _ in range(params.k - L)]
        for x in streams_per_x:
            streams_per_x[x][region] = eval_poly(coeffs, x)
    names = custodians or [f"Custodian {x}" for x in range(1, params.n + 1)]
    return [Share(x=x, params=params, paper_id=paper_id, epoch=0, custodian=names[x - 1],
                  streams=streams_per_x[x]) for x in range(1, params.n + 1)]


# --------------------------------------------------------------------------
# Reconstruction / Reed-Solomon decoding
# --------------------------------------------------------------------------
@dataclass
class RegionDecode:
    data: np.ndarray        # (L, G) recovered secret bytes
    agree: np.ndarray       # (G,) number of shares consistent with the chosen polynomial
    errors: np.ndarray      # (m, G) bool: share j disagrees in group g


@dataclass
class Reconstruction:
    image: np.ndarray
    status: str             # insufficient | unverified | clean | corrected | detected
    xs: list[int]
    k: int
    detect_limit: int       # max bad shares that are always detected
    correct_limit: int      # max bad shares that are always corrected
    share_error_groups: dict[int, int]      # x -> #groups where share x was wrong
    share_error_maps: dict[int, np.ndarray]  # x -> (H, W) bool tamper map
    uncorrectable_map: np.ndarray           # (H, W) bool
    regions: dict[str, RegionDecode]

    @property
    def guilty(self) -> list[int]:
        return [x for x, c in self.share_error_groups.items() if c > 0]


def _decode_stream(xs: list[int], Y: np.ndarray, k: int, L: int) -> RegionDecode:
    """Unique decoding by trying k-subsets (n <= 8 keeps this cheap).

    For every k-subset S we interpolate, predict the other shares and count
    agreements per group. The true polynomial agrees with >= m - e shares;
    any other degree<k polynomial agrees with <= k - 1 + e, so the winner is
    unique whenever e <= (m - k) / 2. Groups that already agree with all m
    shares are finished after the first subset, so clean data costs one pass.
    """
    m, G = Y.shape
    data = np.zeros((L, G), dtype=np.uint8)
    agree_best = np.full(G, -1, dtype=np.int16)
    err_best = np.zeros((m, G), dtype=bool)
    pending = np.arange(G)
    for S in combinations(range(m), k):
        if pending.size == 0:
            break
        Sx = [xs[s] for s in S]
        vinv = gf.vandermonde_inverse(Sx)
        ysub = [Y[s, pending] for s in S]
        coeffs = [gf.combine(vinv[i], ysub) for i in range(k)]
        errs = np.zeros((m, pending.size), dtype=bool)
        for j in range(m):
            if j not in S:
                errs[j] = eval_poly(coeffs, xs[j]) != Y[j, pending]
        agree = (m - errs.sum(axis=0)).astype(np.int16)
        better = agree > agree_best[pending]
        idx = pending[better]
        agree_best[idx] = agree[better]
        err_best[:, idx] = errs[:, better]
        for i in range(L):
            data[i, idx] = coeffs[i][better]
        pending = pending[agree_best[pending] < m]
    return RegionDecode(data=data, agree=agree_best, errors=err_best)


def _interpolate_stream(xs: list[int], Y: np.ndarray, L: int) -> RegionDecode:
    """Plain interpolation through exactly the given points (no checking)."""
    m, G = Y.shape
    vinv = gf.vandermonde_inverse(xs)
    data = np.zeros((L, G), dtype=np.uint8)
    for i in range(min(L, m)):
        data[i] = gf.combine(vinv[i], list(Y))
    return RegionDecode(data=data, agree=np.full(G, m, dtype=np.int16),
                        errors=np.zeros((m, G), dtype=bool))


def check_compatible(shares: list[Share]) -> list[str]:
    """Human-readable warnings about mixing shares (metadata is not authenticated)."""
    warnings = []
    if len({s.paper_id for s in shares}) > 1:
        warnings.append("Shares come from different papers.")
    if len({s.params for s in shares}) > 1:
        raise ValueError("Shares were created with different parameters.")
    if len({s.x for s in shares}) != len(shares):
        raise ValueError("Two shares have the same x-coordinate.")
    return warnings


def reconstruct(shares: list[Share]) -> Reconstruction:
    check_compatible(shares)
    params = shares[0].params
    H, W, C = params.shape
    k, m = params.k, len(shares)
    xs = [s.x for s in shares]
    lay = layout(params)

    flat = np.zeros(H * W * C + 1, dtype=np.uint8)
    groups_bad = {x: 0 for x in xs}
    bad_bytes = {x: np.zeros(H * W * C + 1, dtype=bool) for x in xs}
    unc_bytes = np.zeros(H * W * C + 1, dtype=bool)
    decoded = {}
    any_err = any_unc = False

    for region, pos in lay.items():
        L = params.L(region)
        Y = np.stack([s.streams[region] for s in shares])
        if m > k:
            rd = _decode_stream(xs, Y, k, L)
        else:
            rd = _interpolate_stream(xs, Y, L)
        decoded[region] = rd
        t = (m - k) // 2
        unc = rd.agree < (m - t)
        for i in range(L):
            flat[pos[:, i]] = rd.data[i]
        if m > k:
            ok = ~unc
            for j, x in enumerate(xs):
                e = rd.errors[j] & ok
                groups_bad[x] += int(e.sum())
                bad_bytes[x][pos[e].ravel()] = True
                any_err |= bool(e.any())
            unc_bytes[pos[unc].ravel()] = True
            any_unc |= bool(unc.any())

    def to_pix(b):
        return b[:-1].reshape(H, W, C).any(axis=2)

    if m < k:
        status = "insufficient"
    elif m == k:
        status = "unverified"
    elif any_unc:
        status = "detected"
    elif any_err:
        status = "corrected"
    else:
        status = "clean"

    return Reconstruction(
        image=flat[:-1].reshape(H, W, C), status=status, xs=xs, k=k,
        detect_limit=max(0, m - k), correct_limit=max(0, (m - k) // 2),
        share_error_groups=groups_bad,
        share_error_maps={x: to_pix(b) for x, b in bad_bytes.items()},
        uncorrectable_map=to_pix(unc_bytes), regions=decoded)


# --------------------------------------------------------------------------
# Proactive refresh
# --------------------------------------------------------------------------
def refresh(shares: list[Share]) -> tuple[list[Share], dict[str, bool]]:
    """Add shares of a random polynomial whose L data coefficients are 0.

    Secret is unchanged; old and new shares no longer combine. Regions with
    L == k have no random coefficients, so they cannot be refreshed.
    """
    params = shares[0].params
    new = [s.copy() for s in shares]
    refreshed = {}
    for region in REGIONS:
        L = params.L(region)
        G = shares[0].streams[region].size
        refreshed[region] = L < params.k
        if L == params.k:
            continue
        delta = [np.zeros(G, dtype=np.uint8)] * L + [random_bytes(G) for _ in range(params.k - L)]
        for s in new:
            s.streams[region] ^= eval_poly(delta, s.x)
    for s in new:
        s.epoch += 1
    return new, refreshed


# --------------------------------------------------------------------------
# Attacks (for the demo)
# --------------------------------------------------------------------------
def _groups_touching(params: Params, pix_mask: np.ndarray) -> dict[str, np.ndarray]:
    b = np.concatenate([byte_mask_from_pixels(pix_mask, params.shape[2]), [False]])
    return {r: b[pos].any(axis=1) for r, pos in layout(params).items()}


def rect_mask(shape, rect) -> np.ndarray:
    H, W = shape[:2]
    x0, y0, x1, y1 = rect
    m = np.zeros((H, W), dtype=bool)
    m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    return m


def tamper_random(share: Share, rect) -> Share:
    """Corrupt the share bytes that cover a pixel rectangle."""
    out = share.copy()
    for r, hit in _groups_touching(share.params, rect_mask(share.params.shape, rect)).items():
        noise = random_bytes(int(hit.sum()))
        noise[noise == 0] = 1
        out.streams[r][hit] ^= noise
    return out


def forge_whole(share: Share) -> Share:
    out = share.copy()
    for r in REGIONS:
        out.streams[r] = random_bytes(out.streams[r].size)
    return out


def targeted_forgery(share: Share, coalition_xs: list[int], fake: np.ndarray,
                     assumed_original: int = 255) -> tuple[Share, list[str]]:
    """Tompa-Woll style cheat: shift share so the reconstruction shows `fake`.

    `fake` is an H x W x C image with the attacker's content, and the attacker
    only *guesses* the original is white (255) where they write. Works on
    regions with L == 1 when exactly k shares (coalition_xs) are combined.
    Returns the forged share and the list of regions that could be targeted.
    """
    params = share.params
    out = share.copy()
    vinv = gf.vandermonde_inverse(coalition_xs)
    lam = vinv[0][coalition_xs.index(share.x)]
    flat_fake = np.concatenate([fake.ravel(), [assumed_original]]).astype(np.uint8)
    targeted = []
    for r, pos in layout(params).items():
        if params.L(r) != 1:
            continue
        delta = (flat_fake[pos[:, 0]] ^ np.uint8(assumed_original)).astype(np.uint8)
        out.streams[r] ^= gf.scale(gf.inv(lam), delta)
        targeted.append(r)
    return out, targeted


# --------------------------------------------------------------------------
# Visualisation helpers
# --------------------------------------------------------------------------
def paint(params: Params, values: dict[str, np.ndarray]) -> np.ndarray:
    """Paint one byte per group over all pixels of that group (image-sized)."""
    H, W, C = params.shape
    flat = np.zeros(H * W * C + 1, dtype=np.uint8)
    for r, pos in layout(params).items():
        if r in values:
            flat[pos] = values[r][:, None]
    return flat[:-1].reshape(H, W, C)


def share_to_grid(share: Share) -> np.ndarray:
    return paint(share.params, share.streams)


def collusion_view(shares: list[Share], region: str) -> np.ndarray | None:
    """Best structured leak that len(shares) colluders can compute, or None.

    In a ramp scheme the k-L random coefficients hide everything from up to
    k-L colluders (their shares are uniformly random). With c >= k-L+1
    shares, colluders can cancel the random coefficients and obtain a fixed
    linear combination of the L secret bytes of every group - which shows
    the layout of a document. Returns that combination per group.
    """
    params = shares[0].params
    k, L = params.k, params.L(region)
    need = k - L + 1
    if len(shares) < need:
        return None
    use = shares[:need]
    xs = [s.x for s in use]
    if need == 1:
        w = [1]
    else:
        # Solve sum_{j<need-1} w_j x_j^i = x_last^i for i = L..k-1 (char 2: minus = plus).
        A = [[gf.power(xs[j], i) for j in range(need - 1)] for i in range(L, k)]
        b = [gf.power(xs[-1], i) for i in range(L, k)]
        Ainv = gf.invert_matrix(A)
        w = []
        for j in range(need - 1):
            acc = 0
            for i in range(need - 1):
                acc ^= gf.mul(Ainv[j][i], b[i])
            w.append(acc)
        w.append(1)
    return gf.combine(w, [s.streams[region] for s in use])
