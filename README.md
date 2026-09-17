# DAS workshop

Materials for the hands-on DAS (Deep Audio Segmenter) session. All you need to start is conda.

**Please do the install before the session.** The DAS environment is a big download, and conference
wifi is slow.

Start by cloning this repository (see **Get the files** below). For now it only has this README and
the two setup notebooks, which is all you need before the workshop. Install everything, then run
the setup check. The workshop notebooks and the recording will be added to the repository later,
and you can get them with `git pull`. Until then the setup check will point out that they're
missing, but that's expected and nothing to worry about.

## What's in the workshop

There's a setup check, then three parts that you go through in order:

| | File | What it covers |
|---|---|---|
| **0** | `00_check_your_setup_mac.ipynb`<br>`00_check_your_setup_windows.ipynb` | *Is my machine ready?* Run the one for your computer after installing, before the session. It checks that every package is there, that audio plays, and that you can download the example data. If something's missing, it tells you how to fix it. It doesn't install or change anything. |
| **1** | `01_hearing_fly_song.ipynb` | *What does the data look like?* Listen to pulse and sine song, see how they differ, and let a clustering algorithm sort the pulses into groups on its own. Based on [the DAS tutorial](https://janclemenslab.org/das/unsupervised/flies.html), with extra cells so you can hear the song. |
| **2** | `02_labelling_and_training.ipynb` | *How do I train DAS?* Annotate a recording in the DAS GUI, train a network on it, annotate a second stretch to test it, and let the network label the rest. There's no code in this one. You follow the steps in the DAS window. |
| **3** | `03_looking_at_your_model.ipynb` | *Did it work?* Look at what you annotated, score the network against your own labels, and work out how much that score really tells you. |

You can do part 1 on its own. Parts 2 and 3 need DAS, and part 3 uses the files you make in part 2.

Open every notebook from JupyterLab. Part 2 is just instructions, so keep it open next to the DAS
window while you work.

---

## Get the files

Clone the repository. This creates an `audio-segmentation-workshop` folder in whatever directory
your terminal is in, so `cd` somewhere sensible first, like your Documents folder:

```bash
git clone https://github.com/deutschlab/audio-segmentation-workshop.git
cd audio-segmentation-workshop
```

**No `git`?** On a Mac, the first time you type `git` it offers to install the command line tools.
Say yes, then run the command again. On Windows, install
[Git for Windows](https://git-scm.com/download/win) and open a new terminal.

When the rest of the workshop files are ready, get them with:

```bash
cd audio-segmentation-workshop
git pull
```

Run all the commands below from inside this folder.

---

## Install

Everything goes into one conda environment. It has DAS and everything the notebooks need, so you
don't have to set up anything else.

**1. Get conda** if you don't already have it. Any recent version is fine. Install
[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/), then close and reopen your terminal.

**2. Create the environment.** Mac and Windows need different Python versions, because `das`
installs a different TensorFlow version on each:

```bash
# macOS
conda create -n das -y python=3.11

# Windows: use 3.10, not 3.11. On Windows das installs TensorFlow 2.10, which doesn't support 3.11.
conda create -n das -y python=3.10
```

**3. Install DAS.** Use these exact versions, not the latest ones:

```bash
conda activate das
pip install "das==0.32.11" "xarray-behave==0.37.4"
pip install jupyterlab umap-learn hdbscan
```

The second line installs JupyterLab, which DAS doesn't include, and the two clustering packages
used at the end of part 1. The whole thing takes a few minutes.

> **Don't install TensorFlow yourself.** `das` already picks the right version for your computer.
> If you specify one, pip won't be able to resolve the install on anything but a Mac.
>
> **Don't install the newest DAS either.** Recent versions use PyTorch and need Python 3.12 or
> newer. The notebooks and the GUI steps in part 2 were written for `0.32.11`, and a newer version
> won't match them.

**4. Check that it worked.** This is part 0. Open the setup notebook for your computer and run all
the cells. It checks everything and tells you how to fix anything that fails:

```bash
cd audio-segmentation-workshop
conda activate das
jupyter lab 00_check_your_setup_mac.ipynb        # or ..._windows.ipynb
```

If the last cell says READY, you're all set.

For a quick check instead of the full one:

```bash
python -c "import das; print(das.__version__)"
das gui        # opens the DAS annotation window; just close it again
```

If you get stuck, the official DAS install guide is here:
<https://janclemenslab.org/das/installation.html>

---

## Running part 1

Once `git pull` has brought in the workshop notebooks, activate the environment and start
JupyterLab from **inside the folder**:

```bash
conda activate das
jupyter lab 01_hearing_fly_song.ipynb
```

JupyterLab opens in your browser. Run a cell with **Shift+Enter**, or click the play button next to it.

Because you started JupyterLab from the `das` environment, it should already use the right kernel.
If you want to double-check, run this in a cell:

```python
import sys
print(sys.executable)
```

The path should be inside the `das` environment. If it isn't, go to **Kernel → Change Kernel…** and
choose the right one. When you see "No module named ...", this is nearly always the reason.

## What you should see

Run the cells from top to bottom.

1. **Imports.** Nothing is printed if everything is installed. If you didn't install the two
   clustering packages, you'll see `Clustering section unavailable`. That's only a warning, and
   the rest of the notebook still runs.
2. **Download.** Downloads a 2 MB example recording of a courting male fly and prints what's inside.
3. **The four arrays.** Prints the shape, type and first few values of each one. The audio is
   stored as a single column, and the annotation times are in samples, not seconds.
4. **Loaded.** Prints `sample rate 10000 Hz | recording 112.4 s | 356 marked pulses |
   823 marked sine points`. If you see this, the data loaded correctly.
5. **Pulse.** An audio player, about 9 seconds long. It isn't one continuous bout. It's all the
   stretches that have pulse song and no sine song, joined together with short gaps of silence.
   It prints the peak frequency of the pulses, **349 Hz**. Then it measures each pulse separately,
   and those values spread much wider, because a pulse has a broad spectrum rather than one clear
   pitch.
6. **Sine.** The same thing for sine song, about 5 seconds long. Its peak frequency is **146 Hz**,
   a bit more than an octave below pulse.
7. **Spectrum and spectrogram.** Pulse shows up as a broad bump in the spectrum peaking around
   350 Hz, and as separate vertical lines in the spectrogram. Sine shows up as a narrow peak around
   150 Hz and one long horizontal band. That difference is what the network has to learn.
8. **One pulse up close.** A single pulse, about 12 ms long.
9. **Inter-pulse interval histogram.** Prints `278 gaps measured, typical value 32 ms`, and the
   same value as pulses per second. There's also a smaller bump around 60 ms. That's where a quiet
   pulse was missed, so two gaps got counted as one.
10. **Clustering.** Five cells. They cut out every annotated event, reduce the data with UMAP, show
    the clusters (`3 clusters found`), and compare them with the human annotations. The last cell
    runs the exact same steps on random noise. The noise forms clusters too, and that's the lesson:
    clusters on their own don't prove anything.
11. **The recording you'll annotate.** Loads `Dmel_male.wav` (390 s) and plots the first 18 seconds.

### Can't hear the sine song?

Sine song is a hum at about **150 Hz**, which is too low for most laptop speakers. Headphones solve
this. You can also speed it up by running this in a new cell:

```python
listen(sine_clip, speed=4)
```

This plays the same clip four times faster, so the hum moves up to about 600 Hz. The sound itself
isn't filtered or changed, only the playback speed.

### The clustering section

These five cells cluster the annotated events to look for different types of pulse. They include
the comparison with the human annotations and the noise test. They need `umap` and `hdbscan`,
which step 3 of the install already includes. If you skipped them, install them now:

```bash
conda activate das
pip install umap-learn hdbscan
```

If they aren't installed, those five cells **skip themselves with a message**. The rest of the
notebook still runs, including the last section about the recording you'll annotate in part 2. So
you can install them later if you want.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `Clustering section unavailable` in the first cell | `umap-learn` and `hdbscan` aren't installed. Only the five clustering cells are affected. |
| `No module named 'numpy'` / `'scipy'` / `'matplotlib'` | You're on the wrong kernel. Check `sys.executable` as shown above, then use **Kernel → Change Kernel…** |
| `jupyter: command not found` | The environment isn't activated, or JupyterLab isn't installed in it (install step 3). |
| `conda: command not found` | Conda isn't installed, or you need to close and reopen your terminal after installing it. |
| `git: command not found` | Git isn't installed. See **Get the files** above. |
| `git pull` says your local changes would be overwritten | Running a notebook saves its output into the file, so git thinks you changed it. Run `git stash`, then `git pull` again. |
| JupyterLab can't find `00_check_your_setup_mac.ipynb` | You're not in the cloned folder. Run `cd audio-segmentation-workshop` first. |
| No audio player appears | Some editors, like PyCharm and VS Code, don't show audio players. Open the notebook in JupyterLab in your browser. |
| The download cell hangs | It downloads 2 MB from GitHub. Check your internet connection and run that cell again. |
| Part 3 says "No annotation CSVs" | In the first code cell, set `FOLDER` to the folder you exported from the GUI in part 2. |
| Part 3 says "NOT a held-out score" | You scored the model on audio it was trained on. Run the prediction past the training range and score that part. |
| Part 3 says "Nothing to compare" | Your own annotations and the network's proposals don't cover the same stretch. First check `EXPORT_START_S`. It has to match the start of the range you exported in part 2 step 6, because DAS saves times relative to that start. Then check that you annotated 18-42 s by hand (part 2 step 5), and that you exported the proposals **before** approving them. The table above that cell shows where your annotations actually are. |
| Part 3's numbers look far too low, or the times start near zero | Same cause. `Export for DAS` saves times relative to the start of the exported range, so in an 18-78 s export, a pulse at 20 s is saved as `2.0`. Set `EXPORT_START_S` to the start of your range, and part 3 converts the times back. |
| The interval histogram suddenly stops at 20 ms | That comes from the minimal-distance setting in DAS's threshold detector, not from the fly. It's fine for this recording. Part 2 step 2 shows where the setting is. |
