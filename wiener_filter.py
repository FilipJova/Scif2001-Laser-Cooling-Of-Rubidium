#!/usr/bin/env python3
"""
Isolate the bright (trapped) spot using background subtraction + a
frequency-domain Wiener filter, then suppress everything that is not the spot.

    python wiener_spot.py Trapped.tif Bgd.tif            # -> spot_only.tif, figure
    python wiener_spot.py Trapped.tif Bgd.tif --psf-sigma 2

Pipeline
--------
1. Load both frames (RGBA TIFFs -> single grey channel; alpha is ignored).
2. D = Trapped - Background           removes static scene (dust, rails, glare).
3. Wiener filter on D:
        W(f) = conj(H) S(f) / ( |H|^2 S(f) + N(f) )
   N(f): noise power spectrum, estimated from a spot-free quiet region.
   S(f): signal power spectrum, estimated as  max(<|D(f)|^2> - N, 0)
         (the local average <.> over frequency approximates the expectation).
   H   : optional Gaussian PSF (psf_sigma = 0 -> pure denoising Wiener filter).
4. Detect the spot in the filtered image (k-sigma threshold on the *filtered*
   noise level), keep only the connected component that contains the peak,
   fill holes, and feather the edge.
5. spot_only = filtered * mask  -> exactly zero everywhere else.
"""
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


# --------------------------------------------------------------------------- I/O
def load_gray(path):
    """Read a TIFF and return one float grey channel (R==G==B in these files)."""
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float64)


# ------------------------------------------------------------------ Wiener filter
def gaussian_psf_otf(shape, sigma):
    """Optical transfer function of a Gaussian PSF (sigma in pixels)."""
    if sigma <= 0:
        return np.ones(shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.exp(-2.0 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))


def estimate_noise_sigma(img, spot_xy, exclude=60, quiet_box=None):
    """
    Robust per-pixel noise sigma from a quiet window that excludes the spot.
    quiet_box = (y0, y1, x0, x1); default: central band away from the rails.
    """
    h, w = img.shape
    if quiet_box is None:
        quiet_box = (int(0.30 * h), int(0.60 * h), int(0.15 * w), int(0.80 * w))
    y0, y1, x0, x1 = quiet_box
    win = img[y0:y1, x0:x1].copy()
    m = np.ones_like(win, bool)
    sx, sy = spot_xy
    m[max(0, sy - exclude - y0):max(0, sy + exclude - y0),
      max(0, sx - exclude - x0):max(0, sx + exclude - x0)] = False
    vals = win[m]
    # iterative 3-sigma clip so isolated dust residue does not inflate sigma
    mu, sd = vals.mean(), vals.std()
    for _ in range(5):
        keep = np.abs(vals - mu) < 3 * sd
        mu, sd = vals[keep].mean(), vals[keep].std()
    return sd


def wiener_filter(img, noise_sigma, psf_sigma=0.0, smooth_f=3.0):
    """
    Frequency-domain Wiener filter/deconvolution with white-noise model.

    Using numpy's un-normalised FFT, a white-noise image of variance s^2 has
    E|N(f)|^2 = s^2 * Npix.
    """
    npix = img.size
    F = np.fft.fft2(img)
    H = gaussian_psf_otf(img.shape, psf_sigma)

    P = np.abs(F) ** 2                       # observed power spectrum
    # local average over frequency ~ expectation of the periodogram
    P_avg = ndi.gaussian_filter(P, smooth_f, mode="wrap")
    N = noise_sigma ** 2 * npix * np.ones_like(P)
    S = np.clip(P_avg - N, 1e-12, None) / np.maximum(np.abs(H) ** 2, 1e-6)

    W = np.conj(H) * S / (np.abs(H) ** 2 * S + N)
    out = np.real(np.fft.ifft2(W * F))
    return out, W


# --------------------------------------------------------------- spot extraction
def extract_spot(filt, noise_sigma_filt, k=6.0, feather=2.0, peak_xy=None):
    """Binary + feathered mask of the single component that holds the peak."""
    # smooth a little so the peak search is not fooled by a hot pixel
    sm = ndi.gaussian_filter(filt, 3)
    if peak_xy is None:
        py, px = np.unravel_index(np.argmax(sm), sm.shape)
    else:
        px, py = peak_xy
    binary = filt > k * noise_sigma_filt
    lab, _ = ndi.label(binary)
    lbl = lab[py, px]
    if lbl == 0:                              # peak itself fell below threshold
        raise RuntimeError("No component above threshold at the peak - lower k.")
    mask = ndi.binary_fill_holes(lab == lbl)
    soft = ndi.gaussian_filter(mask.astype(float), feather) if feather > 0 else mask * 1.0
    soft = np.clip(soft, 0, 1) * (ndi.binary_dilation(mask, iterations=int(3 * feather) + 1))
    return mask, soft, (px, py)


# ------------------------------------------------------------------------ main
def run(trapped_path, bgd_path, psf_sigma=0.0, k=6.0, feather=2.0,
        out_tif="spot_only.tif", out_png="wiener_spot_figure.png"):
    T, B = load_gray(trapped_path), load_gray(bgd_path)
    D = T - B                                            # step 2

    # rough peak position from a heavily smoothed difference image
    sm = ndi.gaussian_filter(D, 4)
    py, px = np.unravel_index(np.argmax(sm), sm.shape)

    sigma_n = estimate_noise_sigma(D, (px, py))          # step 3
    filt, W = wiener_filter(D, sigma_n, psf_sigma=psf_sigma)
    filt = np.clip(filt, 0, None)                        # intensity cannot be < 0

    # noise level *after* filtering, measured in the same quiet region
    qs = (slice(int(0.30 * D.shape[0]), int(0.60 * D.shape[0])),
          slice(int(0.15 * D.shape[1]), int(0.80 * D.shape[1])))
    q = filt[qs].copy()
    q[max(0, py - 60 - qs[0].start):max(0, py + 60 - qs[0].start),
      max(0, px - 60 - qs[1].start):max(0, px + 60 - qs[1].start)] = np.nan
    sigma_f = np.nanstd(q)

    mask, soft, (px, py) = extract_spot(filt, sigma_f, k=k, feather=feather,
                                        peak_xy=(px, py))
    spot_only = filt * soft                              # step 5

    # ------------------------------------------------------------- metrics
    peak = spot_only.max()
    cnt_raw = D[mask].sum()
    cnt_out = spot_only.sum()
    ys, xs = np.nonzero(mask)
    print(f"spot peak (x,y)      : ({px}, {py})")
    print(f"noise sigma, raw diff: {sigma_n:.3f} counts")
    print(f"noise sigma, filtered: {sigma_f:.3f} counts  "
          f"(x{sigma_n / sigma_f:.1f} reduction)")
    print(f"peak value           : {peak:.1f} counts   -> SNR(raw) = {D[py, px] / sigma_n:.0f}")
    print(f"spot area            : {mask.sum()} px   bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
    print(f"integrated counts    : raw {cnt_raw:.0f}  vs  output {cnt_out:.0f} "
          f"({100 * cnt_out / cnt_raw:.1f}% retained)")
    print(f"non-zero px outside spot mask: {(spot_only[~ndi.binary_dilation(mask, iterations=8)] != 0).sum()}")

    # --------------------------------------------------------------- outputs
    out16 = np.clip(spot_only / spot_only.max() * 65535, 0, 65535).astype(np.uint16)
    Image.fromarray(out16).save(out_tif)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pad = 70
        sl = (slice(max(0, py - pad), py + pad), slice(max(0, px - pad), px + pad))
        fig, ax = plt.subplots(2, 3, figsize=(15, 9))
        ax = ax.ravel()
        ax[0].imshow(T, cmap="gray", vmax=np.percentile(T, 99.7)); ax[0].set_title("Trapped (raw)")
        ax[1].imshow(D, cmap="gray", vmin=-20, vmax=60); ax[1].set_title("Trapped - Background")
        ax[2].imshow(np.fft.fftshift(W), cmap="viridis")
        ax[2].set_title("Wiener transfer function W(f)")
        ax[3].imshow(D[sl], cmap="gray", vmin=-20, vmax=60); ax[3].set_title("Zoom: difference")
        ax[4].imshow(filt[sl], cmap="gray"); ax[4].set_title("Zoom: Wiener-filtered")
        ax[5].imshow(spot_only[sl], cmap="inferno"); ax[5].set_title("Spot only (background = 0), zoom")
        for a in ax[[0, 1]]:
            a.axis("off")
        plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()
    except ImportError:
        pass
    return spot_only, mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trapped"); ap.add_argument("background")
    ap.add_argument("--psf-sigma", type=float, default=0.0,
                    help="Gaussian PSF sigma (px) for deconvolution; 0 = denoise only")
    ap.add_argument("-k", type=float, default=6.0, help="detection threshold in noise sigmas")
    ap.add_argument("--feather", type=float, default=2.0, help="edge softening (px)")
    ap.add_argument("--out-tif", default="spot_only.tif")
    ap.add_argument("--out-png", default="wiener_spot_figure.png")
    a = ap.parse_args()
    run(a.trapped, a.background, a.psf_sigma, a.k, a.feather, a.out_tif, a.out_png)
