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
python dyniclick/click_detection.py -h
```

## Click detection

```sh
python dyniclick/click_detection.py examples/example.wav examples/example.clicks.pk --bandpass_freq 10000 15000 15000 20000
```

-> the creates a pickle file with the click times and amplitudes

## Click analysis

```sh
python dyniclick/click_analysis.py examples/example.wav examples/example.clicks.pk examples/example.feat.pk --tdoa_max 0.00047
```

Note: tdoa_max is the maximum delay between the hydrophones, i.e. the distance between the hydrophones divided by the speed of sound in water (~ 1500 m/s)


## Click tracking

```sh
python dyniclick/click_tracking.py examples/example.feat.pk examples/example.tracks.pk --click_interval_max 0.3 --diff_max 0.000025 --amp_thres 0.1
```


## Plot click features

```sh
python dyniclick/plot_utils.py examples/example.feat.pk --feat_cols 1 4 2 5 --feat_scale 1 1000 1000 0.001 --track_file examples/example.tracks.pk
```
