# Video Annotator

The Video Annotator allows you to efficiently extract images from videos and send them to Groundlight for annotation and training. Video Annotator uses clustering to find a diverse set of images that will train your Groundlight detector to a high confidence level with minimal annotation effort. 

## Annotating a video

After you set up your environment (see below), you can run the following commands to build a CV model and annotate a video.  Everything is stored in your project directory, so you can pick up where you left off.

First, run `./src/s1setup.py <VIDEO_PATH>` to set up the project directory, and cluster the frames to find diverse representative frames.

Then run `./src/s2train.py <PROJECT_DIRECTORY>` to create a detector, and send the most interesting frames to the model. During this time, you should open the dashboard and label the images as they come in.  This script will wait for confident answers. 
If you want to use an existing detector, or use a different human labeling configuration, there are feature flags available to control the behavior of the script. 

Now run `./src/s3produce.py <PROJECT_DIRECTORY>` to run all the frames through the detector, and build a new video with the results.

## Setting up your dev environment

There are at least a couple ways to set up your development environment.  Pick one that works for you.

### Simple setup (venv, pip)

1. Clone the repository and initialize submodules:
```
git clone <repo-url>
cd video-annotator
git submodule update --init --recursive
```

2. Create virtual environment and install dependencies:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note**: The `shared-ui` submodule contains shared CSS/JS components. Make sure to initialize it with `git submodule update --init --recursive` after cloning.

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

## Troubleshooting

### libGL.so.1: cannot open shared object file: No such file or directory

On Ubuntu, if you get an error about `libGL`, you may need to install the libgl1-mesa-glx package:

```
sudo apt update
sudo apt install -y libgl1-mesa-glx
```
