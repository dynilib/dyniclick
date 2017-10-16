Click detector, Inter Pulse Interval and cross-channel delay computation.

# INSTALL

1 - Clone repository

```sh
$ git clone ssh://git@194.167.251.152:8002/jul/click_detection.git
```

2 - Install [Miniconda](https://conda.io/docs/install/quick.html).

3 - Create environnement:

```sh
$ conda env create -f environment.yml
```

# USAGE EXAMPLES

First, activate the environnement:

```sh
$ source activate click_detector
```

To get help on any script:

```sh
python myscript.py -h
```

Click detection:

```sh
python click_detector.py somefile.wav somefile.clicks --bandpass_freqs 10000 20000 20000 30000 --threshold 0.5 --channel 0 --highpass_freq 10000
```

Click analysis:

```sh
python click_analysis.py somefile.wav somefile.clicks somefile.feat --highpass_freq 10000 --ipi_min 0.0015 --ipi_max 0.008 --delay_max 0.00065 --channels 0 1
```

Plot click features:

```sh
python plot_utils.py somefile.feat --feat_col 1 2 4 5 6 --feat_name 'Click amp' 'IPI (ms)' 'TDAO (ms)' $'Spectrum\nargmax (kHz)' $'Spectral\ncentroid\n(kHz)' --feat_scale 1 1000 1000 0.001 0.001 --feat_thres 0.1 0 0 0 0
```

Add click visualization on top of a video

```sh
python video_click.py somefile.mp4 somefile.feat output.mp4 --max_tdoa 0.00065 --cols 0 1 3
```
