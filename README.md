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

Click detection:

```sh
python click_detector.py someaudiofile.wav someaudiofile.clicks --cutoff_freqs 10000 20000 20000 30000 --threshold 0.5 --channel 0 --show 1
```

Click IPIs and cross-channel delay:

```sh
python ipi_xchanneldelay_extractor.py someaudiofile.wav someaudiofile.clicks someaudiofile.delays --cutoff_freq 10000 --ipi_min 0.0015 --ipi_max 0.008 --delay_max 0.0015 --channels 0 1
```

Plot IPIs and cross-channel delays:

```sh
python plot_utils.py someaudiofile.delays
```
