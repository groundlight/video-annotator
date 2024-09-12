# Video Annotator

## Annotating a video

After you set up your environment (see below), you can run the following commands to build a CV model and annotate a video.  Everything is stored in your project directory, so you can pick up where you left off.

First, run `s1setup.py` to set up the project directory, and cluster the frames to find the most diverse frames.

Then run `s2train.py` to create a detector, and send the most interesting frames to the model.  During this time, you should open the dashboard and label the images as they come in.  This script will wait for confident scores.

Now run `s3infer.py` to run all the frames through the detector, and build a new video with the results.

## Setting up your dev environment

### Simple setup (venv)

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
```

### Advanced setup (conda and direnv)

```
conda create -n video-annotator python=3.11
sudo apt update && apt install direnv
echo 'eval "$(direnv hook bash)"'  >> ~/.bashrc
echo 'export CONDA_BASE_PATH="$(dirname "$(dirname "$(which conda)")")"' >> ~/.bashrc
direnv allow .
conda activate video-annotator  # Should be automatic and unneeded
poetry install
```
