# %%
from matplotlib.cm import binary
from matplotlib.cm import binary
import numpy as np
from  pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from PIL import Image
from scipy import ndimage as ndi
import csv

# %%
import cv2  # still used to save images out
import os
import numpy as np
from decord import VideoReader
from decord import cpu, gpu

def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using decord's VideoReader
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    # load the VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)
                     
    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = len(vr)

    frames_list = list(range(start, end, every))
    saved_count = 0

    if every > 25 and len(frames_list) < 1000:  # this is faster for every > 25 frames and can fit in memory
        frames = vr.get_batch(frames_list).asnumpy()

        for index, frame in zip(frames_list, frames):  # lets loop through the frames until the end
            save_path = os.path.join(frames_dir, video_filename, "{:010d}.tif".format(index))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # save the extracted image
                saved_count += 1  # increment our counter by one

    else:  # this is faster for every <25 and consumes small memory
        for index in range(start, end):  # lets loop through the frames until the end
            frame = vr[index]  # read an image from the capture
            
            if index % every == 0:  # if this is a frame we want to write out based on the 'every' argument
                save_path = os.path.join(frames_dir, video_filename, "{:010d}.tif".format(index))  # create the save path
                if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                    cv2.imwrite(save_path, cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))  # save the extracted image
                    saved_count += 1  # increment our counter by one

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=False, every=1):
    """
    Extracts the frames from a video
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible
    
    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)
    
    print("Extracting frames from {}".format(video_filename))
    
    extract_frames(video_path, frames_dir, every=every)  # let's now extract the frames

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames


# %%
#script_dir = Path(__file__).parent.absolute()
#file_path = script_dir.parent.parent / 'Video' / 'Decay 1_2.mp4'

#if __name__ == '__main__':
#    video_to_frames(video_path=file_path, frames_dir='Frames', overwrite=False, every=100)

# %% 
# 
#import sys
#sys.path.insert(0, str(script_dir.parent))  # add the scif1 folder to the path so we can import wiener_filter.py
#import wiener_filter as ws

#script_dir = Path(__file__).parent.absolute()
#file_path = script_dir.parent.parent / 'Video' / '0.tif'
#background_path = script_dir.parent.parent / 'Video' / 'Decay1_Background.tif'

#ws.run(trapped_path=file_path, bgd_path=background_path, psf_sigma=0.0, k=3.0, feather=2.0,
#        out_tif="spot_only.tif", out_png="wiener_spot_figure.png")

# %%

import sys
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir.parent))  # add the scif1 folder to the path so we can import wiener_filter.py
import wiener_filter as ws

def process_frame(path, bgd_path, psf_sigma=0.0, peak_xy=None):
    """Read peak at fixed position. peak_xy=(px, py) from first frame."""
    img = ws.load_gray(path)
    bgd = ws.load_gray(bgd_path)
    D = img - bgd

    sigma_n = ws.estimate_noise_sigma(D, (0, 0))
    if not np.isfinite(sigma_n) or sigma_n < 1e-6:
        sigma_n = 1e-6

    filt, _ = ws.wiener_filter(D, sigma_n, psf_sigma=psf_sigma)
    filt = np.nan_to_num(filt, nan=0.0, posinf=0.0, neginf=0.0)
    filt = np.clip(filt, 0, None)

    py, px = peak_xy
    patch = filt[py-2:py+3, px-2:px+3]
    val = patch.max()
    return 0.0 if not np.isfinite(val) else val


if __name__ == '__main__':
    video_path = str(script_dir.parent.parent / 'Video' / 'Decay 1_2.mp4')
    background_path = str(script_dir.parent.parent / 'Video' / 'Decay1_Background.tif')
    frames_dir = video_to_frames(video_path, 'Frames', overwrite=False, every=60)

    frame_files = sorted(os.listdir(frames_dir))

    # find peak position from the FIRST (brightest) frame
    first = ws.load_gray(os.path.join(frames_dir, frame_files[0]))
    bgd = ws.load_gray(background_path)
    D0 = first - bgd
    sm = ndi.gaussian_filter(D0, 4)
    py, px = np.unravel_index(np.argmax(sm), sm.shape)
    print(f"Fixed peak position: ({px}, {py})")

    results = []
    for i, fname in enumerate(frame_files):
        fpath = os.path.join(frames_dir, fname)
        peak = process_frame(fpath, background_path, peak_xy=(py, px))
        results.append((fname, peak))
        print(f"{fname}: peak = {peak:.2f}")

    csv_path = "peak_luminosity(6f)2.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "peak_luminosity"])
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {csv_path}")
# %%
na0 = 5.24*(10^6)

t = np.linspace(1,100)

tau = na0*1/np.e

N = na0*np.exp(-t/tau)

# N = nao/np.e

plt.figure()
plt.title("Atom Count")
plt.plot(t, N,)
plt.xlabel("Time(sec)")
plt.ylabel("Atom count")
plt.legend()
plt.grid(True)
plt.show()