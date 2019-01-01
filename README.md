# v2s
automatic foley for short videos

We modify WaveGAN architecture (Donahue et al.) to be a conditional generative model. The goal is to produce natural audio conditioned on video priors. 

To train a model, store checkpoints and predictions:
```
python train.py train ./train --data_dir drumming
```

predictions.ipynb is used for creating wav files from the predicted numpy arrays.

TODO: Need to resolve batch loading issue. This seems to be preventing proper training.
