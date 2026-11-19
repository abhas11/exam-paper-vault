# Exam Paper Vault

Demo for the thesis *"A Size-Efficient, Cheater-Identifying Threshold Image Sharing
Scheme for Secure Examination Paper Distribution"*.

Thesis draft (method, security analysis, results, chapter plan): [docs/THESIS_DRAFT.md](docs/THESIS_DRAFT.md)

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q tests
```

## What's inside

| File | Purpose |
|---|---|
| `sss_exam/gf256.py` | GF(2^8) arithmetic with lookup tables (same field as `galois.GF(2**8)`) |
| `sss_exam/scheme.py` | ROI-adaptive ramp sharing, Reed–Solomon cheater identification, refresh, attacks, share file format |
| `sss_exam/metrics.py` | Entropy, adjacent-pixel correlation, NPCR/UACI, PSNR |
| `sss_exam/sample.py` | Synthetic exam paper and the attacker's fake notice |
| `app.py` | Streamlit demo (6 tabs) |
| `tests/test_scheme.py` | Correctness tests |

## Demo script for a presentation

1. **Paper & split**: show the question region and the storage table (Shamir 5× vs this config ≈ 3.9×). Click Split.
2. **Shares**: each custodian's share looks like noise.
3. **Exam morning**:
   - 2 custodians → noise (below threshold).
   - 3 custodians, one set to *Targeted forgery* → a convincing fake "Q5 cancelled" notice and **no warning**.
   - Add the other 2 → *"Cheater identified: Exam Board"*, tamper map, and the genuine paper.
4. **Leakage study**: one Thien–Lin share (L = k) shows the whole paper; L = 1 and the ramp L = 2 stay noise.
5. **Refresh**: refresh, then in tab 3 use *Old share (stolen before refresh)*: it's rejected.
6. **Metrics**: tables for the results chapter.

## Design notes

- Each group of L bytes is the first L coefficients of a degree-(k−1) polynomial; the rest are random.
  L = 1 is Shamir, L = k is Thien–Lin, in between is a ramp scheme (any k−L shares reveal nothing).
- Shares are a Reed–Solomon codeword. The decoder tries k-subsets (fine for n ≤ 8) and picks the
  polynomial most shares agree with. It corrects ⌊(m−k)/2⌋ bad shares and detects m−k.
  Berlekamp–Welch is the scalable alternative for larger n.
- Share metadata (x, n, k, L, ROI, paper id) is stored unauthenticated in a PNG text chunk.
- Not done yet: hierarchical custodians (at least one Exam Board member required).
