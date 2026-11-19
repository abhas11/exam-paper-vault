# A Size-Efficient, Cheater-Identifying Threshold Image Sharing Scheme for Secure Examination Paper Distribution

*Working draft: outline, method and first results. Numbers come from the demo in this repository
(`app.py`) on the built-in sample paper (720 × 1040 RGB) with n = 5, k = 3.*

---

## Abstract (draft)

Question-paper leaks happen when a single insider (printer, clerk or courier) gets the full paper before
the exam. We propose a threshold image sharing system: the paper image is split into *n* shares held by
different custodians. Any *k* of them can rebuild it on exam morning, while *k − 1* learn nothing about the
questions. Building on Shamir's secret sharing over GF(2⁸), the scheme adds three things:

1. **ROI-adaptive ramp sharing.** The question region is shared with perfect secrecy (L = 1) and the
   public header with a ramp scheme (L > 1), so shares are smaller than plain Shamir.
2. **Cheater identification.** Because shares form a Reed–Solomon codeword, the system detects forged
   or tampered shares, names the cheater, repairs the damage and draws a pixel-level tamper map.
3. **Proactive refresh.** Shares can be re-randomized so that stolen old shares become useless.

On the sample paper the scheme uses 22 % less storage than plain Shamir. Recovery is lossless and shares
are statistically random (entropy 8.00 bits/byte, adjacent-pixel correlation ≈ 0). With 5 shares present
it identifies and corrects one cheater, including a Tompa–Woll style forgery that goes undetected when
exactly k shares are used.

---

## 1. Introduction

### 1.1 Problem
- Exam papers pass through many hands before the exam (setters, printers, transport, centres).
- One leak means cancelled exams, re-tests and loss of trust.
- Encryption alone moves the problem to *who holds the key*. Trusting a single key holder is the same
  weakness again.

### 1.2 Idea
Split the paper so that **no single person (or small group) can open it**, but **any k authorised custodians
together can**, without a trusted key holder.

### 1.3 Threat model

| Threat | Requirement |
|---|---|
| One insider leaks their share | Reveals nothing |
| Up to k − 1 insiders collude | Reveal nothing about the questions |
| A custodian loses a share or does not turn up | Paper is still recoverable from any k |
| A custodian submits a forged or tampered share | Detected; cheater named and damage corrected when enough shares are present |
| Tompa–Woll cheat: cheater makes others see a fake paper and learns the real one | Prevented by verification with more than k shares |
| A share is stolen; a custodian is transferred | Refresh makes old shares useless |

Out of scope: a dishonest dealer at creation time, side channels, time-lock release.

### 1.4 Contributions
1. An ROI-adaptive ramp image sharing scheme over GF(2⁸), with a tunable storage/secrecy trade-off per region.
2. Reed–Solomon cheater identification applied to image shares, giving per-pixel tamper localisation.
3. A demonstration that plain Thien–Lin sharing leaks the layout of text documents, with an exact
   characterisation of what *c* colluders learn for each L.
4. Proactive refresh for image shares, and the observation that L = k regions cannot be refreshed.
5. An open-source implementation and an interactive demo, evaluated with standard image-sharing metrics.

---

## 2. Mathematical background

### 2.1 Finite field GF(2⁸)
- Elements are bytes 0–255, so every pixel value is a field element. Unlike the original Thien–Lin
  scheme, which works mod 251, no pixel clipping is needed.
- Addition = XOR. Multiplication is modulo the irreducible polynomial x⁸ + x⁴ + x³ + x² + 1 (0x11D),
  the same default field as the `galois` Python library.
- Implementation: log/antilog tables and a 256 × 256 multiplication table. Multiplying a whole image
  array by a constant is then a single table lookup (`sss_exam/gf256.py`).

### 2.2 Shamir's (k, n) threshold scheme
- Secret s; random a₁ … a_{k−1}; f(x) = s + a₁x + … + a_{k−1}x^{k−1}.
- Share i = (xᵢ, f(xᵢ)) for distinct non-zero xᵢ (we use x = 1 … n).
- Any k shares give f, and hence s = f(0), by Lagrange interpolation:
  s = Σⱼ yⱼ · Πₘ≠ⱼ xₘ / (xₘ − xⱼ).
- With k − 1 shares, every s is equally likely (perfect secrecy).
- Worked example to include: the GF(2⁸) polynomial 83x² + 5x + 12 from the early experiments, and the
  prime-field example with p = 11.

### 2.3 Applying it to images (naïve approach)
- Flatten the H × W × C image to bytes and share every byte independently with fresh randomness.
- Measured on the sample paper: each share = 2,193.8 KB (100 %), and 5 shares = **5.00×** storage.

### 2.4 Thien–Lin (2002)
- Put k pixels into all k coefficients, with no randomness. Each share is 1/k the size (1.67× for n = 5, k = 3).
- Weakness: identical pixel groups give identical share bytes, so structure leaks (Section 5).

### 2.5 Ramp schemes (Blakley–Meadows 1984)
- L secret coefficients + (k − L) random ones.
- Any k − L shares reveal nothing; k shares reveal everything; in between, partial information.

### 2.6 Reed–Solomon codes
- (f(x₁), …, f(xₙ)) for deg f < k is a Reed–Solomon codeword of length n and dimension k
  (McEliece & Sarwate 1981).
- Minimum distance n − k + 1. With m received shares you can detect up to m − k errors and correct up
  to ⌊(m − k)/2⌋.

---

## 3. Related work (to expand)
- Secret sharing: Shamir 1979; Blakley 1979.
- Visual cryptography: Naor & Shamir 1994. Needs no computation, but has pixel expansion and contrast loss.
- Secret image sharing: Thien & Lin 2002 and follow-ups (meaningful shadows, progressive, essential shares).
- Cheating detection: Tompa & Woll 1988; McEliece & Sarwate 1981; Berlekamp–Welch decoding.
- Proactive secret sharing: Herzberg, Jarecki, Krawczyk & Yung 1995.
- Hierarchical secret sharing: Tassa 2007 (future work).
- Computational alternative: Krawczyk 1993, "Secret sharing made short" (AES + share the key). This is
  the main baseline for the discussion.

---

## 4. Proposed scheme

### 4.1 System overview
```
 Exam board                         Custodians                    Exam centre (exam morning)
 ┌───────────┐  split  ┌──── share x=1  Exam Board ────┐
 │ question  │───────▶ │──── share x=2  District Officer│──▶ any k shares ──▶ RS decode ──▶ paper
 │ paper img │         │──── share x=3  Centre Supdt.   │      (more = verified,           + verdict
 └───────────┘         │──── share x=4  Police Observer │       cheater named)           + tamper map
                       └──── share x=5  Secure Server ──┘
```

### 4.2 Regions and layout
- The image bytes are split into two regions: **ROI** (the question area, a rectangle) and **background**
  (header, logo, instructions).
- Each region is cut, in raster order, into groups of L_region consecutive bytes, zero-padded at the end.
- The layout is a deterministic function of (shape, ROI, L values). Only the ROI *position* is public,
  never its content.

### 4.3 Share generation
For each group g with secret bytes d₀ … d_{L−1}:

f_g(x) = d₀ + d₁x + … + d_{L−1}x^{L−1} + r_L x^L + … + r_{k−1}x^{k−1}, with fresh random rᵢ.

Share x stores f_g(x) for every group (one byte per group).

- Share size = ⌈|ROI| / L_roi⌉ + ⌈|BG| / L_bg⌉ bytes.
- Randomness comes from a CSPRNG (Python `secrets`), not `numpy.random`.
- File format: the share bytes as a grayscale PNG, plus a metadata text chunk
  (x, n, k, L values, image shape, ROI, paper id, epoch).

### 4.4 Reconstruction and cheater identification
Input: m shares.
- **m < k:** cannot reconstruct. The demo shows what such a coalition gets: noise.
- **m = k:** interpolate directly. The result is *unverified*: a cheater goes unnoticed.
- **m > k:** unique decoding by k-subset search.
  1. For each k-subset S, interpolate the coefficients and predict the other m − k shares.
  2. Count, per group, how many shares agree.
  3. Keep the polynomial with the most agreement.
  4. Groups where all m agree are finished after the first subset, so clean data costs one pass.
- **Why it is unique:** the true polynomial agrees with ≥ m − e shares. Any other polynomial of
  degree < k agrees with the true one on ≤ k − 1 points, so it agrees with the received shares on
  ≤ k − 1 + e. These ranges are disjoint when e ≤ ⌊(m − k)/2⌋.
- **Outputs:**
  - Per share: the number of groups where it disagreed, giving the verdict "cheated" or "consistent"
    and a tamper map.
  - Groups with too many errors are flagged as uncorrectable.
- **Cost:** C(m, k) subsets, which is cheap for n ≤ 8. Berlekamp–Welch is the scalable alternative
  for larger n.

| Shares present (k = 3) | Result |
|---|---|
| 2 | noise (below threshold) |
| 3 | rebuilt but unverifiable |
| 4 | tampering detected, cheater not identifiable |
| 5 | one cheater identified **and** corrected |

### 4.5 Targeted forgery (Tompa–Woll), as attack and test case
- For a group with L = 1 and a known reconstruction set S, d₀ = Σⱼ λⱼ yⱼ with Lagrange weights λⱼ.
- The cheater adds δ = Δ / λᵢ to their own share, which shifts the output by exactly Δ.
- To write a fake notice, the attacker guesses that the page is white (255) there and sets
  Δ = fake ⊕ 255.
- Result with exactly k shares: a convincing "NOTICE: Q5 is CANCELLED" appears (PSNR 19.8 dB against
  the original, with no warning). The cheater can subtract Δ and read the real paper.
- With 5 shares: the system names the cheater and restores the genuine paper (PSNR ∞).
- For L > 1 the attacker controls only one byte per group, which shifts L pixels by fixed multiples,
  so a targeted forgery of all L pixels is not possible.

### 4.6 Proactive refresh
- Every share adds δ(xᵢ), where δ has **zero** data coefficients and random higher coefficients.
- The secret is unchanged and no one reconstructs the paper during the refresh.
- Old and new shares are no longer consistent. Combining an old share with new ones gives garbage,
  or the decoder flags the old share as a cheater.
- Regions with L = k have no random coefficients, so they **cannot be refreshed**. This is another
  argument against plain Thien–Lin.

---

## 5. Security analysis

### 5.1 Secrecy against colluders (ramp bound)
- With c ≤ k − L shares, the map from the random coefficients to the c share values is a bijection
  (an x^L-scaled Vandermonde system). The shares are therefore uniformly random and independent of
  the secret.
- With c ≥ k − L + 1 shares, colluders can find weights w with Σⱼ wⱼxⱼⁱ = 0 for i = L … k − 1. This
  cancels the randomness and reveals Σᵢ (Σⱼ wⱼxⱼⁱ) dᵢ, a fixed linear combination of the secret bytes.
- **Experiment (Leakage tab):**

| L | 1 colluder | 2 colluders |
|---|---|---|
| 1 (Shamir) | noise | noise |
| 2 (ramp) | noise | layout visible |
| 3 (Thien–Lin) | **paper fully readable** | readable |

For x = 1 and L = 3, a white group (255, 255, 255) maps to 255 ⊕ 255 ⊕ 255 = 255, so white stays white
and the text stays black.
- Conclusion: use L = 1 for the questions, and L > 1 only for content that is public anyway.

### 5.2 Integrity
- Detection and correction bounds as in 4.4. They hold for any modification of a share, not only
  random noise.
- Limitation: share metadata is not authenticated. A stale share is still caught by the decoder,
  because it is inconsistent with the others.

### 5.3 What the scheme does *not* give
- No protection against a malicious dealer (that would need verifiable secret sharing, e.g. Feldman
  or Pedersen).
- With exactly k shares there is no verification. The protocol must require at least k + 2 custodians.

---

## 6. Implementation

| Module | Role |
|---|---|
| `sss_exam/gf256.py` | GF(2⁸) tables, Lagrange weights, Vandermonde inverse |
| `sss_exam/scheme.py` | layout, split, reconstruct and RS decode, refresh, attacks, collusion view, share I/O |
| `sss_exam/metrics.py` | entropy, adjacent correlation, NPCR/UACI, PSNR, PNG size |
| `sss_exam/sample.py` | synthetic exam paper and fake notice |
| `app.py` | Streamlit demo, 6 tabs |
| `tests/test_scheme.py` | 16 tests: field axioms, all k-subsets, cheater ID, forgery, refresh, collusion, file round-trip |

Development path (earlier scripts):
1. `lang.py`: Lagrange interpolation mod p.
2. `finite.py`: Shamir over GF(2⁸) with `galois`.
3. `imageTomatrix.py`: image ↔ byte array.
4. `shadow.py` / `fromShadow.py`: naïve (3, 5) image sharing, measuring the 5× overhead.
5. This repository: table-based GF(2⁸), ramp + ROI, RS decoding, refresh, demo.

Performance (720 × 1040 × 3 paper, n = 5, k = 3, laptop CPU):

| Operation | Time |
|---|---|
| Split into 5 shares | ≈ 0.2–0.6 s |
| Reconstruct from exactly 3 shares | ≈ 0.2 s |
| Verify all 5 (clean) | ≈ 0.8 s |
| Identify and correct 1 cheater, one region tampered | ≈ 0.7 s |
| Identify and correct 1 cheater, whole page tampered | ≈ 3.5 s |

---

## 7. Results (sample paper, n = 5, k = 3, L_roi = 1, L_bg = 2)

### 7.1 Storage

| Scheme | Each share | All 5 shares |
|---|---|---|
| Plain Shamir (L = 1) | 2,193.8 KB | 5.00× |
| Thien–Lin (L = 3) | 731.2 KB | 1.67× (leaks) |
| **Proposed (ROI L = 1, BG L = 2)** | **1,708.2 KB** | **3.89×, 22 % less than Shamir** |
| AES + Shamir on the key | 32 B + ciphertext | ≈ 1× (computational security only) |

### 7.2 Share statistics

| Image | Entropy (bits/byte) | Adjacent corr. | NPCR % | UACI % |
|---|---|---|---|---|
| Original | 0.90 | 0.78 | – | – |
| Shares x = 1 … 5 | 8.00 | −0.018 … 0.001 | 99.61 | 48.9 |

- UACI is measured against a mostly white page, where ≈ 50 % is expected. The commonly quoted 33.5 %
  applies between two random images. State the reference explicitly.
- PNG does not compress the shares (≈ raw size), while the original compresses to 84 KB.

### 7.3 Cheating experiments
- Targeted forgery, 3 shares: fake notice accepted silently (PSNR 19.8 dB).
- Same forgery, 5 shares: "Cheater identified: Exam Board (x = 1)", output identical to the original.
- Stale share after refresh, 5 shares: flagged as the cheater.
- Random tampering of the Q5 region in one share: identified, localised on the tamper map and corrected.

### 7.4 Still to measure
- Detection and correction rates over many random trials and varying numbers of bad shares.
- Time vs image size, n and k (plots).
- Comparison with published schemes (table of features: expansion, secrecy, cheater ID, refresh).

---

## 8. Limitations and future work
- **Hierarchical access:** require at least one Exam Board custodian (Tassa 2007). Note that Birkhoff
  interpolation uses derivatives, which mostly vanish in characteristic 2. This needs a prime field
  (e.g. GF(257)) or Hasse derivatives.
- Berlekamp–Welch decoding for large n.
- Authenticated share metadata (MAC or signature by the board).
- Verifiable secret sharing against a dishonest dealer.
- Smarter ROI (auto-detect question blocks) without leaking the layout through the mask.
- Time-release: shares usable only after the exam start time.

---

## 9. Proposed chapter plan (≈ 75–100 pages)

| Ch. | Title | Pages |
|---|---|---|
| 1 | Introduction | 5–7 |
| 2 | Mathematical background | 12–18 |
| 3 | Literature review | 10–14 |
| 4 | Proposed scheme | 15–20 |
| 5 | Security analysis | 6–10 |
| 6 | Implementation | 6–10 |
| 7 | Results and evaluation | 12–18 |
| 8 | Conclusion and future work | 3–4 |
| – | References, appendices | 8–15 |

---

## References (check each against the original before the final version)
1. A. Shamir, "How to share a secret," *Communications of the ACM*, 22(11), 612–613, 1979.
2. G. R. Blakley, "Safeguarding cryptographic keys," *Proc. AFIPS National Computer Conference*, 1979.
3. G. R. Blakley and C. Meadows, "Security of ramp schemes," *CRYPTO '84*, LNCS 196, 1985.
4. R. J. McEliece and D. V. Sarwate, "On sharing secrets and Reed–Solomon codes," *Communications of the ACM*, 24(9), 583–584, 1981.
5. M. Tompa and H. Woll, "How to share a secret with cheaters," *Journal of Cryptology*, 1(2), 133–138, 1988.
6. M. Naor and A. Shamir, "Visual cryptography," *EUROCRYPT '94*, LNCS 950, 1995.
7. C.-C. Thien and J.-C. Lin, "Secret image sharing," *Computers & Graphics*, 26(5), 765–770, 2002.
8. A. Herzberg, S. Jarecki, H. Krawczyk and M. Yung, "Proactive secret sharing or: How to cope with perpetual leakage," *CRYPTO '95*, LNCS 963, 1995.
9. H. Krawczyk, "Secret sharing made short," *CRYPTO '93*, LNCS 773, 1994.
10. T. Tassa, "Hierarchical threshold secret sharing," *Journal of Cryptology*, 20(2), 237–264, 2007.
11. L. R. Welch and E. R. Berlekamp, "Error correction for algebraic block codes," US Patent 4,633,470, 1986.
12. Y. Wu, J. P. Noonan and S. Agaian, "NPCR and UACI randomness tests for image encryption," *Cyber Journals: JSAT*, 2011.
