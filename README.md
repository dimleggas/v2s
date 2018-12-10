# v2s
automatic foley for short videos

We modify WaveGAN architecture (Donahue et al.) to be a conditional generative model. The goal is to produce natural audio conditioned on video priors. See report for full details.

tfrecord.py: Saves audio and video features from numpy to tfrecords.

loader.py: Responsible for loading data batches from tfrecords.

wavegan.py: The conditional wavegan.

train.py: Train a model; store checkpoints and predictions. The usage is
```
python train.py train ./train --data_dir drumming
```

predictions.ipynb: Used for creating wav files from the predicted numpy arrays.