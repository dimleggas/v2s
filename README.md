# v2s
automatic foley for short videos

We modify WaveGAN architecture (Donahue et al.) to be a conditional generative model. The goal is to produce natural audio conditioned on video priors. See report for full details.

vid_feat_extractor.py: Preprocesses videos and extracts video features using the output of the first fully-connected layer of VGG19 for feature representation.

aud_feat_extractor.py: Contains functions we used to preprocess audios and a function for generating wav files from generated numpy arrays.

tfrecord.py: Saves audio and video features from numpy to tfrecords.

loader.py: Responsible for loading data batches from tfrecords.

wavegan.py: The conditional wavegan.

train.py: Train a model; store checkpoints and predictions. The usage is
```
python train.py train ./train --data_dir drumming
```

predictions.ipynb: Used for creating wav files from the predicted numpy arrays.

# Results

Real Snoring: https://www.youtube.com/watch?v=J6frJNO0eZE

Generated Snoring: https://www.youtube.com/watch?v=RVU-ovRxjpg

Real Drum: https://www.youtube.com/watch?v=x0lxlaxJlCs

Generated Drum: https://www.youtube.com/watch?v=5d47iurVfZY

