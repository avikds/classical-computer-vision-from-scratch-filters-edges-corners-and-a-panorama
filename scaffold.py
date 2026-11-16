"""
Classical Computer Vision from Scratch: Filters, Edges, Corners and a Panorama scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Classical Computer Vision from Scratch: filters, edges, corners and a panorama.

Runs every stage you built on eight generated scene pairs and measures it against the truth:
separable blur, Canny under noise, Harris versus Shi-Tomasi repeatability, sub-pixel refinement,
Hartley normalization, the ratio test, RANSAC against the plain DLT and against its iteration formula,
and finally exposure matching and blending, scored by PSNR against the true scene."""
import time
import numpy as np

from solution import (convolve2d, gaussian_kernel1d, gaussian_blur, sobel_gradients, non_max_suppression,
                      hysteresis_threshold, canny, structure_tensor, corner_response, local_maxima, refine_subpixel,
                      harris_corners, apply_homography, normalize_points, dlt_homography, transfer_error,
                      describe_patches, match_descriptors, corner_repeatability, ransac_num_iterations,
                      ransac_homography, homography_corner_error, bilinear_sample, warp_image, panorama_canvas,
                      feather_weights, match_exposure, stitch_panorama)

# The scene generator (the same code the steps use): a flat world of shapes seen by two cameras.


def _world_shapes(seed):
    """A seeded flat 'world': a smooth background plus layered polygons and discs (world units ~ pixels)."""
    r = np.random.RandomState(7919 + 31 * seed)
    bg = [(r.uniform(-0.06, 0.06), r.uniform(60, 220), r.uniform(0, 2 * np.pi), r.uniform(0, 2 * np.pi)) for _ in range(4)]
    shapes = []
    for _ in range(240):
        kind = r.choice(4, p=[0.35, 0.2, 0.3, 0.15])
        cx, cy = r.uniform(50, 410), r.uniform(15, 225)
        size = r.uniform(4, 18)
        val = r.uniform(0.05, 0.95)
        if kind == 3:
            shapes.append(("disc", (cx, cy, size * 0.7), val))
            continue
        if kind == 0:
            a, b = size * r.uniform(0.4, 1.0), size * r.uniform(0.4, 1.0)
            th = r.uniform(0, np.pi)
            loc = np.array([[-a, -b], [a, -b], [a, b], [-a, b]])
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            V = loc @ R.T
        else:
            n = 3 if kind == 1 else 4
            ang = np.sort(r.uniform(0, 2 * np.pi, n))
            while np.max(np.diff(np.r_[ang, ang[0] + 2 * np.pi])) > 0.75 * 2 * np.pi or np.min(np.diff(np.r_[ang, ang[0] + 2 * np.pi])) < 0.6:
                ang = np.sort(r.uniform(0, 2 * np.pi, n))
            rad = size * r.uniform(0.6, 1.0, n)
            V = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        shapes.append(("poly", V + np.array([cx, cy]), val))
    return bg, shapes


def _eval_world(seed, X, Y):
    bg, shapes = _world_shapes(seed)
    img = 0.5 + 0.0 * X
    for amp, lam, ph, ang in bg:
        img = img + amp * np.sin(2 * np.pi * (X * np.cos(ang) + Y * np.sin(ang)) / lam + ph)
    for kind, g, val in shapes:
        if kind == "disc":
            cx, cy, rad = g
            sd = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - rad
        else:
            V = g
            sd = np.full(X.shape, -np.inf)
            for i in range(len(V)):
                p, q = V[i], V[(i + 1) % len(V)]
                e = q - p
                nrm = np.array([e[1], -e[0]]) / np.hypot(e[0], e[1])
                sd = np.maximum(sd, (X - p[0]) * nrm[0] + (Y - p[1]) * nrm[1])
        a = np.clip(0.5 - sd, 0.0, 1.0)
        img = img * (1 - a) + val * a
    return img


def render_world(seed, M, shape):
    """Noise-free image of world `seed`: output pixel (x, y) shows world point M @ (x, y, 1)."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    P = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    Q = np.asarray(M, float) @ P
    return _eval_world(seed, (Q[0] / Q[2]).reshape(h, w), (Q[1] / Q[2]).reshape(h, w))


def _view_to_world(r, cx, cy, h, w):
    th = r.uniform(-0.08, 0.08)
    s = r.uniform(0.95, 1.05)
    A = np.array([[s * np.cos(th), -s * np.sin(th), cx], [s * np.sin(th), s * np.cos(th), cy], [0.0, 0.0, 1.0]])
    P = np.array([[1.0, 0, 0], [0, 1.0, 0], [r.uniform(-4e-4, 4e-4), r.uniform(-4e-4, 4e-4), 1.0]])
    C = np.array([[1.0, 0, -(w - 1) / 2], [0, 1.0, -(h - 1) / 2], [0, 0, 1.0]])
    return A @ P @ C


def make_scene(seed, noise=0.01, shape=(160, 220), shift=100.0):
    """Two overlapping grayscale views of world `seed` and the true homography between them."""
    h, w = shape
    r = np.random.RandomState(104729 + 17 * seed)
    cx, cy = 230 + r.uniform(-15, 15), 120 + r.uniform(-10, 10)
    W1 = _view_to_world(r, cx - shift / 2, cy + r.uniform(-6, 6), h, w)
    W2 = _view_to_world(r, cx + shift / 2, cy + r.uniform(-6, 6), h, w)
    img1 = render_world(seed, W1, shape) + noise * r.randn(h, w)
    gain, offset = r.uniform(0.78, 0.9), r.uniform(0.03, 0.08)
    img2 = gain * render_world(seed, W2, shape) + offset + noise * r.randn(h, w)
    H = np.linalg.inv(W2) @ W1
    H = H / H[2, 2]
    return {"img1": img1, "img2": img2, "H": H, "to_world1": W1, "exposure2": (gain, offset), "seed": seed}


def psnr(x, y, mask):
    return 10 * np.log10(1.0 / ((x - y)[mask] ** 2).mean())


def seam_jump(pano, truth, va, vb):
    """Mean extra brightness step where image 1's footprint ends inside image 2 (per row, right edge)."""
    jumps = []
    for r in range(pano.shape[0]):
        cols = np.nonzero(va[r])[0]
        if len(cols) == 0:
            continue
        c = cols[-1]
        if 2 <= c < pano.shape[1] - 2 and vb[r, c - 2:c + 3].all():
            jumps.append(abs((pano[r, c + 1] - pano[r, c]) - (truth[r, c + 1] - truth[r, c])))
    return float(np.mean(jumps))


def matched_points(I1, I2, ratio=0.8, subpixel=True, sigma=None):
    p1, p2 = harris_corners(I1, subpixel=subpixel), harris_corners(I2, subpixel=subpixel)
    d1, k1 = describe_patches(I1, p1, sigma=sigma)
    d2, k2 = describe_patches(I2, p2, sigma=sigma)
    q1, q2 = p1[k1], p2[k2]
    m = match_descriptors(d1, d2, ratio)
    return q1[m[:, 0]], q2[m[:, 1]]


def main():
    t0 = time.time()
    SEEDS = range(8)
    scenes = {s: make_scene(s) for s in SEEDS}
    shape = scenes[0]["img1"].shape
    print(f"Eight generated scenes: two {shape[0]} x {shape[1]} views each, noise 0.01, view 2 darker by a hidden gain.")

    print("\n== Part 1. Convolution and Gaussian blur ==")
    rng = np.random.default_rng(0)
    big = rng.uniform(size=(256, 256))
    g = gaussian_kernel1d(3.0)
    t = time.time(); a = gaussian_blur(big, 3.0); t_sep = time.time() - t
    t = time.time(); b = convolve2d(big, np.outer(g, g)); t_2d = time.time() - t
    print(f"  sigma 3 on 256 x 256: two 1-D passes {t_sep * 1000:.0f} ms, one 2-D pass {t_2d * 1000:.0f} ms "
          f"({len(g)} vs {len(g) ** 2} taps), max difference {abs(a - b).max():.1e}")
    noise = rng.normal(size=(256, 256))
    print("  noise std after blur (measured / predicted sum g^2):",
          {s: (round(float(gaussian_blur(noise, s)[20:-20, 20:-20].std()), 3), round(float((gaussian_kernel1d(s) ** 2).sum()), 3)) for s in (0.5, 1.0, 2.0, 4.0)})

    print("\n== Part 2. Canny ==")
    sc = scenes[0]
    clean = render_world(0, sc["to_world1"], shape)
    ref = canny(clean)
    print(f"  noise-free view 1: {int(ref.sum())} edge pixels")
    for nz in (0.01, 0.03, 0.06):
        noisy = clean + nz * np.random.default_rng(1).normal(size=shape)
        row = []
        for s in (0.5, 1.0, 2.0):
            e = canny(noisy, sigma=s)
            near = np.zeros_like(ref)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near |= np.roll(np.roll(ref, dr, 0), dc, 1)
            row.append(f"sigma {s}: {int(e.sum())} px, {100 * (e & near).sum() / max(e.sum(), 1):.0f}% on a clean edge")
        print(f"  noise {nz}: " + " | ".join(row))

    print("\n== Part 3. Harris corners ==")
    for nz in (0.01, 0.03, 0.06):
        rep = {m: [] for m in ("harris", "shi-tomasi")}
        for s in range(4):
            scn = scenes[s] if nz == 0.01 else make_scene(s, noise=nz)
            for m in rep:
                p1, p2 = harris_corners(scn["img1"], method=m), harris_corners(scn["img2"], method=m)
                rep[m].append(corner_repeatability(p1, p2, scn["H"], shape))
        print(f"  noise {nz}: repeatability Harris {np.mean(rep['harris']):.2f}, Shi-Tomasi {np.mean(rep['shi-tomasi']):.2f} (4 scenes)")
    for sub in (False, True):
        res = []
        for s in SEEDS:
            a, b = matched_points(scenes[s]["img1"], scenes[s]["img2"], subpixel=sub)
            e = transfer_error(scenes[s]["H"], a, b)
            res.append(np.median(e[e < 2]))
        print(f"  sub-pixel refinement {'on ' if sub else 'off'}: median error of correct matches under the true H {np.mean(res):.2f} px")

    print("\n== Part 4. Homographies and the DLT ==")
    for off in (0, 1000):
        eu, en = [], []
        for s in range(10):
            H0 = scenes[s]["H"] if s in scenes else make_scene(s)["H"]
            r = np.random.default_rng(500 + s)
            p1 = r.uniform([0, 0], [219, 159], size=(30, 2)) + off
            S = np.array([[1.0, 0, off], [0, 1.0, off], [0, 0, 1.0]])
            Ht = S @ H0 @ np.linalg.inv(S)
            p2 = apply_homography(Ht, p1) + 0.5 * r.normal(size=(30, 2))
            c = np.array([[0, 0], [219, 0], [219, 159], [0, 159]], float) + off
            err = lambda H: float(np.sqrt(((apply_homography(H, c) - apply_homography(Ht, c)) ** 2).sum(1)).mean())
            eu.append(err(dlt_homography(p1, p2, normalize=False)))
            en.append(err(dlt_homography(p1, p2)))
        print(f"  coordinates + {off:4d}, 30 points, 0.5 px noise: corner error median / max: plain DLT {np.median(eu):.2f} / {max(eu):.1f} px, normalized {np.median(en):.2f} / {max(en):.2f} px")

    print("\n== Part 5. Describing and matching ==")
    for ratio in (0.6, 0.7, 0.8, 0.9, 1.01):
        n, prec = [], []
        for s in SEEDS:
            a, b = matched_points(scenes[s]["img1"], scenes[s]["img2"], ratio)
            ok = transfer_error(scenes[s]["H"], a, b) < 2
            n.append(len(a)); prec.append(ok.mean())
        print(f"  ratio {ratio:4.2f}: {np.mean(n):5.1f} matches per pair, {100 * np.mean(prec):3.0f}% correct, {np.mean(n) * np.mean(prec):5.1f} correct")
    for sig in (0.0, 2.0):
        n, prec = [], []
        for s in SEEDS:
            a, b = matched_points(scenes[s]["img1"], scenes[s]["img2"], sigma=sig)
            ok = transfer_error(scenes[s]["H"], a, b) < 2
            n.append(ok.sum()); prec.append(ok.mean())
        print(f"  descriptor blur sigma {sig}: {np.mean(n):.1f} correct matches, {100 * np.mean(prec):.0f}% precision")

    print("\n== Part 6. RANSAC ==")
    rows = []
    for s in SEEDS:
        a, b = matched_points(scenes[s]["img1"], scenes[s]["img2"])
        Hd = dlt_homography(a, b)
        Hr, inl, trials = ransac_homography(a, b, rng=np.random.default_rng(1000 + s))
        rows.append((homography_corner_error(Hd, scenes[s]["H"], shape), homography_corner_error(Hr, scenes[s]["H"], shape), inl.mean(), trials))
    rows = np.array(rows)
    print(f"  DLT on all matches: median corner error {np.median(rows[:, 0]):.0f} px")
    print(f"  RANSAC: corner errors {np.round(rows[:, 1], 2).tolist()} px, inlier fraction {rows[:, 2].mean():.2f}, samples drawn {rows[:, 3].astype(int).tolist()}")
    sc = scenes[0]
    r = np.random.default_rng(9)
    n = 100
    p1 = r.uniform([0, 0], [219, 159], size=(n, 2))
    p2 = apply_homography(sc["H"], p1) + 0.3 * r.normal(size=(n, 2))
    bad = r.permutation(n)[:50]
    p2[bad] = r.uniform([0, 0], [219, 159], size=(50, 2))
    N = ransac_num_iterations(0.99, 0.5, 4)
    for iters in (N // 4, N):
        ok = 0
        for t in range(100):
            H4 = ransac_homography(p1, p2, 1.0, max_iters=iters, p=0.999999, rng=np.random.default_rng(2000 + t))[0]
            ok += homography_corner_error(H4, sc["H"], shape) < 1.0
        print(f"  50% outliers, {iters:3d} samples (formula says {N} for 99%): within 1 px in {ok}% of 100 runs, formula predicts {100 * (1 - (1 - 0.5 ** 4) ** iters):.0f}%")

    print("\n== Part 7. The panorama ==")
    table = {}
    for s in SEEDS:
        scn = scenes[s]
        for comp in (False, True):
            for blend in ("overwrite", "average", "feather"):
                res = stitch_panorama(scn["img1"], scn["img2"], blend, comp, rng=np.random.default_rng(3000 + s))
                truth = render_world(s, scn["to_world1"] @ np.linalg.inv(res["T"]), res["panorama"].shape)
                va = warp_image(np.ones(shape), res["T"], res["panorama"].shape)[1]
                vb = res["valid"] & (res["overlap"] | ~va)
                table.setdefault((comp, blend), []).append((psnr(res["panorama"], truth, res["valid"]),
                                                             seam_jump(res["panorama"], truth, va, vb),
                                                             homography_corner_error(res["H"], scn["H"], shape)))
    for (comp, blend), v in table.items():
        v = np.array(v)
        print(f"  exposure matching {'on ' if comp else 'off'}, {blend:9s}: PSNR vs true scene {v[:, 0].mean():.2f} dB, seam jump {v[:, 1].mean():.3f}")
    errs = np.array(table[(True, "feather")])[:, 2]
    print(f"  homography corner error over the 8 pairs: median {np.median(errs):.2f} px, worst {errs.max():.2f} px")
    print(f"\ntotal {time.time() - t0:.1f}s")


main()

