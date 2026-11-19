import itertools

import numpy as np
import pytest

from sss_exam import gf256 as gf
from sss_exam import scheme as sc
from sss_exam.sample import exam_paper, fake_notice


def test_field_axioms():
    a = np.arange(256)
    assert all(gf.mul(x, gf.inv(x)) == 1 for x in range(1, 256))
    assert (gf.MUL == gf.MUL.T).all()
    # distributivity on a sample
    rng = np.random.default_rng(1)
    for _ in range(200):
        x, y, z = rng.integers(0, 256, 3)
        assert gf.mul(x, y ^ z) == gf.mul(x, y) ^ gf.mul(x, z)
    assert gf.mul(2, 128) == 0x1D  # reduction by 0x11D


def test_matches_galois_if_installed():
    galois = pytest.importorskip("galois")
    GF = galois.GF(2 ** 8)
    a = GF(np.arange(256))
    assert (np.array(a[:, None] * a[None, :]) == gf.MUL).all()


def _params(img, n=5, k=3, L_roi=1, L_bg=2, roi=(10, 20, 50, 60)):
    return sc.Params(n=n, k=k, L_roi=L_roi, L_bg=L_bg, shape=img.shape, roi=roi)


@pytest.mark.parametrize("L_roi,L_bg", [(1, 1), (1, 3), (2, 3), (3, 3), (1, 2)])
def test_any_k_subset_recovers(L_roi, L_bg):
    img = np.random.default_rng(0).integers(0, 256, (37, 61, 3), dtype=np.uint8)
    shares = sc.split(img, _params(img, L_roi=L_roi, L_bg=L_bg))
    for subset in itertools.combinations(shares, 3):
        rec = sc.reconstruct(list(subset))
        assert rec.status == "unverified"
        assert np.array_equal(rec.image, img)


def test_share_sizes():
    img = np.zeros((30, 30, 1), dtype=np.uint8)
    p = sc.Params(n=5, k=3, L_roi=1, L_bg=3, shape=img.shape, roi=None)
    s = sc.split(img, p)[0]
    assert s.nbytes == 300  # 900 bytes / L=3


def test_clean_with_all_shares():
    img = exam_paper(200, 260)
    shares = sc.split(img, _params(img))
    rec = sc.reconstruct(shares)
    assert rec.status == "clean" and np.array_equal(rec.image, img)


def test_identify_and_correct_one_cheater():
    img = exam_paper(200, 260)
    shares = sc.split(img, _params(img))
    shares[3] = sc.tamper_random(shares[3], (5, 5, 150, 100))
    rec = sc.reconstruct(shares)
    assert rec.status == "corrected"
    assert rec.guilty == [4]
    assert np.array_equal(rec.image, img)
    assert rec.share_error_maps[4][50, 50] and not rec.share_error_maps[4][250, 190]


def test_detect_with_k_plus_one():
    img = exam_paper(200, 260)
    shares = sc.split(img, _params(img))
    bad = sc.forge_whole(shares[1])
    rec = sc.reconstruct([shares[0], bad, shares[2], shares[4]])
    assert rec.status == "detected"


def test_targeted_forgery_with_exactly_k():
    img = exam_paper(240, 320)
    p = _params(img, roi=(0, 120, 240, 320))
    shares = sc.split(img, p)
    rect = (20, 140, 220, 200)
    fake = fake_notice(img.shape, rect)
    forged, targeted = sc.targeted_forgery(shares[0], [1, 2, 3], fake)
    assert "roi" in targeted
    rec = sc.reconstruct([forged, shares[1], shares[2]])
    assert rec.status == "unverified"
    roi_fake = fake[140:200, 20:220]
    roi_true = img[140:200, 20:220]
    white = (roi_true == 255)
    # where the paper was white, the attacker's text shows up exactly
    assert np.array_equal(rec.image[140:200, 20:220][white], roi_fake[white])
    # with a 4th honest share the cheat is identified and undone
    rec2 = sc.reconstruct([forged, shares[1], shares[2], shares[3], shares[4]])
    assert rec2.status == "corrected" and rec2.guilty == [1]
    assert np.array_equal(rec2.image, img)


def test_refresh():
    img = exam_paper(160, 200)
    p = _params(img, L_roi=1, L_bg=2)
    old = sc.split(img, p)
    new, refreshed = sc.refresh(old)
    assert refreshed == {"bg": True, "roi": True}
    assert np.array_equal(sc.reconstruct(new[:3]).image, img)
    mixed = sc.reconstruct([old[0], new[1], new[2]])
    assert not np.array_equal(mixed.image, img)
    rec = sc.reconstruct([old[0]] + new[1:])
    assert rec.status == "corrected" and rec.guilty == [1]


@pytest.mark.parametrize("L", [1, 2, 3])
def test_collusion_view_cancels_randomness(L):
    img = exam_paper(120, 150)
    p = sc.Params(n=5, k=3, L_roi=1, L_bg=L, shape=img.shape, roi=None)
    a, b = sc.split(img, p), sc.split(img, p)  # same image, fresh randomness
    for c in range(1, 4):
        va = sc.collusion_view(a[:c], "bg")
        if c < 3 - L + 1:
            assert va is None
        else:
            vb = sc.collusion_view(b[:c], "bg")
            assert np.array_equal(va, vb)  # depends only on the secret


def test_png_roundtrip():
    img = exam_paper(120, 150)
    s = sc.split(img, _params(img))[2]
    s2 = sc.Share.from_png_bytes(s.to_png_bytes())
    assert s2.x == s.x and s2.params == s.params and s2.paper_id == s.paper_id
    for r in sc.REGIONS:
        assert np.array_equal(s.streams[r], s2.streams[r])
