DyniClick : open-source toolbox for stereo audio click detection, analysis, tracking and exploration

# INSTALL

1 - Clone repository

```sh
$ git clone https://github.com/dynilib/dyniclick.git
```

2 - Install [Miniconda](https://conda.io/docs/install/quick.html).

3 - Create environnement:

```sh
$ conda env create -f environment.yml
```

# USAGE EXAMPLES

First, activate the environnement:

```sh
$ source activate dyniclick
```


To get help on any script, use '-h' option, e.g.:

```sh
python click_detection.py -h
```

Click detection:

```sh
python click_detection.py input.wav output.clicks.pk --bandpass_freqs 5000 10000 15000 20000
```

Click analysis:

```sh
python click_analysis.py input.wav input.clicks.pk output.feat.pk --tdoa_max 0.0012
```

Note: tdoa_max is the maximum delay between the hydrophones, i.e. the distance between the hydrophones divided by the speed of sound in water (~ 1500 m/s)


Click tracking:

```sh
python click_tracking.py input.feat.pk output.tracks.pk --click_interval_max 0.3 --diff_max 0.000025 --amp_thres 0.1
```


Plot click features:

```sh
python dyniclick/plot_utils.py test.feat.pk --feat_cols 1 2 4 5 --feat_scale 1 1000 1000 0.001 --track_file input.tracks.pk
```
