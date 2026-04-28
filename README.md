# caltools

Detector characterization tools for FITS-based astronomical imaging.

## Overview

`caltools` provides typed Python routines for common detector calibration and characterization work: master calibration frames, read-noise and fixed-pattern structure, dark current, photon-transfer gain measurements, linearity, and flat-field response. The routines operate on image arrays and FITS metadata, leaving detector-specific acquisition details to the calling workflow. Hardware-specific checks are kept in explicitly named functions.

The library covers bias, dark, and flat master frame generation; read-noise and fixed-pattern noise mapping; dark-current analysis; photon transfer curves and gain measurement; linearity testing; flat-field response characterization; and diagnostic plotting.

## Modules

```
_types.py       SensorConfig, AnalysisResult, Frame/FrameCube type aliases
io.py           FITS I/O, cube loading, header-based config, file grouping
stacking.py     Master bias, dark, flat (median/mean stacking)
stats.py        Welford accumulator, MAD sigma, outlier masking, normality tests
noise.py        Read noise maps, DSNU, FPN, row/column noise, unstable-pixel detection
dark.py         Dark current vs exposure/temperature, spatial structure, warm pixels
gain.py         Photon transfer curve (with/without read noise), full well, noise decomposition
linearity.py    Linearity test and residual error characterization
prnu.py         Photo-response non-uniformity mapping
plotting.py     Diagnostic plots (image maps, histograms, PTC, summary tables)
```

## Installation

Install from source:

```bash
git clone https://github.com/ev-watson/caltools.git
cd caltools
pip install -e .
```

## Quick Start

```python
import caltools as ct

# Build detector configuration from a FITS header
config = ct.sensor_config_from_header("frame.fit", gain=0.5)

# Generate master calibration frames
bias = ct.master_bias(bias_paths)
dark = ct.master_dark(dark_paths, bias=bias)
flat = ct.master_flat(flat_paths, bias=bias, dark=dark)

# Read noise map
rn_map, ts_map = ct.read_noise_map(bias_cube)

# Photon transfer curve and gain
ptc = ct.photon_transfer_curve(flat_groups, bias, config)

# Linearity
lin = ct.linearity_test(flat_paths_by_exposure, bias=bias, config=config)
```

Summary-producing characterization functions return `AnalysisResult` dataclasses with scalar summaries, maps, and metadata. Loading and master-frame helpers return NumPy arrays.

## Requirements

- Python 3.10+
- numpy, scipy, matplotlib
- astropy (FITS I/O)

## License

MIT
