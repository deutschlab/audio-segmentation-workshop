# DAS workshop

Hands-on material for the DAS session. The install below starts from nothing but conda.

**Do the install before the session.** The DAS environment downloads a lot and is slow on
conference wifi.

**Start by cloning the repository,** as described in **Get the files** below. At first it holds only
this file and the two setup notebooks, and that is everything you need for now. Do the install
below, then run part 0. The workshop notebooks and the recording are added to the same repository
later, and part 0 tells you so rather than reporting them as a problem. When they arrive, one
`git pull` fetches them.

## What the workshop is made of

A setup check, then three parts in order. Each answers one question. The install comes first. The
setup check is how you confirm it worked, not a way to start without it.

| | File | The question |
|---|---|---|
| **0** | `00_check_your_setup_mac.ipynb`<br>`00_check_your_setup_windows.ipynb` | *Is my machine ready?* Once the install below is done, run the one for your platform, and do it **before the session**. It checks every package the three parts need, that audio plays, and that the download works, then prints the fix for anything missing. Nothing else in the folder changes anything on your machine, and neither does this. |
| **1** | `01_hearing_fly_song.ipynb` | *What is this data?* Listen to pulse and sine, see why they differ, and let a computer sort clicks into groups unsupervised. Adapted from [the DAS tutorial](https://janclemenslab.org/das/unsupervised/flies.html), with cells added so you can hear the song. |
| **2** | `02_labelling_and_training.ipynb` | *How do I teach the tool?* Mark up a recording in the DAS GUI, train a network on it, mark a second stretch to test it on, and let it label the rest. Reading only, no code to run: DAS does this in its own window. |
| **3** | `03_looking_at_your_model.ipynb` | *Did it work?* Count what you marked, score the network against your own labels, and find out how much that score is worth. |

Part 1 stands alone. Parts 2 and 3 need DAS installed, and part 3 needs the files part 2 produces.

**Do the install first, then run part 0.** Part 0 opens in JupyterLab from inside the environment,
so it has nothing to check until that environment exists. It is step 4 of the install below. If you
have already installed, go straight to it: READY means you are done, and anything else names the
step that fixes it.

Open all three the same way, from JupyterLab. Part 2 has no code in it. It is a set of instructions
to follow while the DAS window is open beside you.

---

## Get the files

Clone the workshop repository. This makes a folder called `audio-segmentation-workshop` wherever
your terminal is, so move somewhere you own first, such as your Documents folder:

```bash
git clone https://github.com/deutschlab/audio-segmentation-workshop.git
cd audio-segmentation-workshop
```

**If `git` is not found:** on macOS, running `git` for the first time offers to install the
command-line tools, so accept and run the line again. On Windows, install
[Git for Windows](https://git-scm.com/download/win) and reopen your terminal.

**Later, to get the workshop notebooks and the recording,** go into the same folder and pull:

```bash
cd audio-segmentation-workshop
git pull
```

Every command below is run from inside this folder.

---

## Install

Everything runs in one conda environment. It gives you DAS itself as well as the notebooks, so
there is no second environment to set up.

**1. Get conda,** if you do not have it. Any recent version works. Install
[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and reopen your terminal.

**2. Create the environment.** The Python version differs by platform, because `das` pins a
different TensorFlow on each and those pins do not both have wheels for the same Python:

```bash
# macOS
conda create -n das -y python=3.11

# Windows. 3.10, not 3.11. das pins TensorFlow 2.10 here, which has no 3.11 wheel.
conda create -n das -y python=3.10
```

**3. Install DAS, pinned.** These exact versions, not "the latest":

```bash
conda activate das
pip install "das==0.32.11" "xarray-behave==0.37.4"
pip install jupyterlab umap-learn hdbscan
```

> **Do not add a TensorFlow pin of your own.** `das` picks the right one per platform, and naming
> one yourself makes the install unresolvable on anything but macOS.
>
> **And do not install the current DAS.** Upstream has since moved to a PyTorch backend and now
> needs Python 3.12 or newer. The workshop notebooks, and the GUI walkthrough in part 2, are written against the
> TensorFlow-era `0.32.11`, so a newer DAS will not match what you are reading.

The DAS environment does not ship with JupyterLab, which is why it is in the line above, together
with the two clustering packages part 1's last section needs.

This takes several minutes.

**4. Check it worked.** This is part 0. Run the notebook for your platform. It checks everything
at once and tells you what to do about anything that fails:

```bash
cd audio-segmentation-workshop
conda activate das
jupyter lab 00_check_your_setup_mac.ipynb        # or ..._windows.ipynb
```

Or, if you only want a quick sign of life rather than the full check:

```bash
python -c "import das; print(das.__version__)"
das gui        # opens the DAS annotation GUI, then close it again
```

Official DAS install page, if anything goes wrong:
<https://janclemenslab.org/das/installation.html>

---

## Running part 1

This section is for when you have the full folder, after `git pull` has fetched the workshop
notebooks. Activate the environment, then from **inside that folder**:

```bash
conda activate das
jupyter lab 01_hearing_fly_song.ipynb
```

It opens in your browser. Run a cell with **Shift+Enter**, or use the play button in the cell's
gutter.

Because you launched JupyterLab from the environment you installed, the default kernel is already
the right one. If you ever want to be sure, run this in a cell:

```python
import sys
print(sys.executable)
```

The path it prints should be inside the environment you just activated. If it is not, use
**Kernel → Change Kernel…** and pick the one that is. Almost every "no module named ..." error in a
notebook is this and nothing else.

## What you should see

Run the cells in order.

1. **Imports.** No output if everything is installed. If you skipped the two clustering packages
   it prints `Clustering section unavailable`, which is a notice rather than an error.
2. **Download.** Fetches a 2 MB example recording of a courting fly, then prints the keys it holds.
3. **The four arrays.** Prints the shape and type of each one, with its first few values. The sound
   is a table one column wide, and the two lists of marks count samples rather than seconds.
4. **Loaded.** Prints `sample rate 10000 Hz | recording 112.4 s | 356 marked pulses |
   823 marked sine points`. This is your check that the data arrived.
5. **Pulse.** An audio player, about 9 s. Not one continuous bout: it is the stretches that hold
   pulse and no sine, joined with a short silence between them. It prints the loudest frequency of
   the clicks, **349 Hz**, and then the range you get measuring one click at a time, which is much
   wider because a pulse spectrum is a hump rather than a line.
6. **Sine.** The same for sine, about 5 s, and its loudest frequency, **146 Hz**, a little over an
   octave lower.
7. **Spectrum, then spectrogram.** Pulse as a broad hump peaking near 350 Hz and a row of separate
   spikes in time. Sine as one narrow spike near 150 Hz and one continuous band. That difference is
   the whole classification problem.
8. **One pulse, close up.** A single click over about 12 ms.
9. **The interval histogram.** Prints `278 gaps measured, typical value 32 ms`, then the same
   figure as clicks per second, with a second smaller bump near 60 ms where a quiet click was
   missed.
10. **Clustering.** Five cells: cut out every marked event, flatten with UMAP, look at the groups
   (`3 clusters found`), check them against what a person marked, then run the identical settings
   on pure noise as a control. Noise makes clumps too, which is the point.
11. **The recording you label next.** Reads `Dmel_male.wav` (390 s) and plots its first 18 seconds.

### If you cannot hear the sine

Sine song is a hum at roughly **150 Hz**, near the bottom of what a laptop speaker can reproduce.
Headphones are the easy fix. If you still want more, run this in a new cell:

```python
listen(sine_clip, speed=4)
```

That plays the same audio four times faster, lifting it to about 600 Hz. Nothing is filtered. Only
the playback rate changes.

### The clustering section

Five cells cluster the marked events to look for pulse sub-types, including the check against the
human marks and the noise control. They need `umap` and `hdbscan`.
Step 3 of the install already includes both. If you left them out, add them now:

```bash
conda activate das
pip install umap-learn hdbscan
```

Without them those five cells **skip themselves with a message** and the rest of the notebook still
runs, including the final section on the recording you annotate in part 2. So you can leave this
until you want it.

---

## Troubleshooting

| What you see | What it means |
|---|---|
| `Clustering section unavailable` in the first cell | `umap-learn` and `hdbscan` are not installed. Everything except the five clustering cells still runs. |
| `No module named 'numpy'` / `'scipy'` / `'matplotlib'` | Wrong kernel. Check `sys.executable` as above, then **Kernel → Change Kernel…** |
| `jupyter: command not found` | The environment is not activated, or JupyterLab is not installed in it (install step 3). |
| `conda: command not found` | Conda is not installed, or the terminal needs reopening after installing it. |
| `git: command not found` | Git is not installed. See **Get the files** above. |
| `git pull` says your local changes would be overwritten | Running a notebook saves its outputs into the file, so git sees it as changed. Run `git stash`, then `git pull` again. |
| `00_check_your_setup_mac.ipynb` not found when starting JupyterLab | You are not inside the cloned folder. `cd audio-segmentation-workshop` first. |
| The audio player does not appear | If you are in PyCharm or VS Code, open the notebook in JupyterLab in a browser instead. Some editors do not render audio widgets. |
| The download cell hangs | It fetches 2 MB from GitHub. Check your connection, then re-run just that cell. |
| Part 3 says "No annotation CSVs" | Set `FOLDER` in its first code cell to the folder you exported from the GUI in part 2. |
| Part 3 says "NOT a held-out score" | You scored the model on audio it was trained on. Predict past the training range and score there. |
| Part 3 says "Nothing to compare" | Your hand marks and the network's proposals do not cover the same stretch. Check `EXPORT_START_S` first: it has to match the start of the range you exported in part 2 step 6, because DAS writes times counted from there. Then check that you hand-marked 18-42 s as part 2 step 5 asks, and that you exported the proposals **before** approving them. The table above it shows where your marks actually are. |
| Part 3's numbers look far too low, or its times start near zero | Same cause. `Export for DAS` writes every time relative to the start of the range you exported, so an 18-78 s export puts a pulse from 20 s at `2.0`. Set `EXPORT_START_S` to that start and part 3 converts everything back to recording time. |
| The interval histogram stops dead at 20 ms | The thresholding detector's minimal-distance setting, not the fly. It is safe for this recording, and part 2 step 2 says where to find it. |
