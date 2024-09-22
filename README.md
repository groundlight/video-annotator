# Video Annotator

## Annotating a video

After you set up your environment (see below), you can run the following commands to build a CV model and annotate a video.  Everything is stored in your project directory, so you can pick up where you left off.

First, run `./src/s1setup.py` to set up the project directory, and cluster the frames to find diverse representative frames.

Then run `./src/s2train.py` to create a detector, and send the most interesting frames to the model.  During this time, you should open the dashboard and label the images as they come in.  This script will wait for confident scores.

Now run `./src/s3infer.py` to run all the frames through the detector, and build a new video with the results.

## Setting up your dev environment

There are at least a couple ways to set up your development environment.  Pick one that works for you.

### Simple setup (venv, pip)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Advanced setup (conda, direnv, poetry)

```
conda create -n video-annotator python=3.11
sudo apt update && apt install direnv
echo 'eval "$(direnv hook bash)"'  >> ~/.bashrc
echo 'export CONDA_BASE_PATH="$(dirname "$(dirname "$(which conda)")")"' >> ~/.bashrc
direnv allow .
conda activate video-annotator  # Should be automatic and unneeded
poetry install
```
