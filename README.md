# caltools

CMOS detector characterization library for astronomical imaging, following EMVA-1288 v4.0 and established sCMOS methods.

## Overview

caltools provides a modular, typed Python toolkit for the full calibration and characterization pipeline of scientific CMOS detectors. It was developed for the QHY268M (IMX571) but is sensor-agnostic — any FITS-based imaging system can use it.

The library covers bias, dark, and flat master frame generation; read noise and fixed-pattern noise mapping; dark current analysis; photon transfer curves and gain measurement; linearity testing; and PRNU characterization — all with publication-quality diagnostic plotting.

## Modules

```
_types.py       SensorConfig, AnalysisResult, Frame/FrameCube type aliases
io.py           FITS I/O, cube loading, header-based config, file grouping
stacking.py     Master bias, dark, flat (sigma-clipped median stacking)
stats.py        Welford accumulator, MAD sigma, outlier masking, gaussianity tests
noise.py        Read noise maps, DSNU, FPN, row/column noise, RTN detection
dark.py         Dark current vs exposure/temperature, spatial structure, warm pixels
gain.py         Photon transfer curve (with/without read noise), full well, noise decomposition
linearity.py    Linearity test and residual error characterization
prnu.py         Photo-response non-uniformity mapping
plotting.py     Publication-quality diagnostic plots (image maps, histograms, PTC, summary tables)
```

## Installation

```bash
pip install caltools
```

Or install from source:

```bash
git clone https://github.com/ev-watson/caltools.git
cd caltools
pip install -e .
```

## Quick Start

```python
import caltools as ct

# Build sensor config from a FITS header
config = ct.sensor_config_from_header("frame.fit", gain=0.5)

# Generate master calibration frames
bias = ct.master_bias(bias_paths)
dark = ct.master_dark(dark_paths, bias=bias)
flat = ct.master_flat(flat_paths, bias=bias, dark=dark)

# Read noise map
rn_map, ts_map = ct.read_noise_map(bias_cube)

# Photon transfer curve and gain
ptc = ct.photon_transfer_curve(flat_pairs, bias)

# Linearity
lin = ct.linearity_test(flat_paths_by_exposure, bias=bias)
```

All analysis functions return `AnalysisResult` dataclasses with scalar summaries, 2-D maps, and metadata.

## Requirements

- Python 3.10+
- numpy, scipy, matplotlib
- astropy (FITS I/O)

## AI Disclosure

AI-assisted tools (Claude, Anthropic) were used during development of this library for code architecture, implementation, and documentation.

## License

MIT
