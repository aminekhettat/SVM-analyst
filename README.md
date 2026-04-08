# SVM Analyst

[![CI](https://github.com/aminekhettat/SVM-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/aminekhettat/SVM-analyst/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/aminekhettat/SVM-analyst)](https://github.com/aminekhettat/SVM-analyst/releases/latest)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/aminekhettat/SVM-analyst)](LICENSE)

Educational simulator for PWM and space-vector modulations focused on PMSM drives.

## Features

- Simulates multiple modulation techniques from the provided thesis PDF:
  - THIPWM 1/6 and 1/4
  - Space vector modulation (SVM)
  - Discontinuous PWM (DPWM): 120°, 60° variants, 30° (DPWM3)
- Interactive GUI with real-time waveform and FFT visualization
- Display options:
  - Line voltages (terminal-to-ground, 0..VBATT) vs phase voltages (terminal-to-terminal, +/-VBATT)
  - PWM waveform vs filtered (fundamental) waveform
  - Optional switching-edge markers for PWM transitions
  - SVM sector highlighting and active sector indicator
- Realistic voltage scaling:
  - Inverter terminal voltages are shown between 0 V and battery voltage (VBATT)
  - Phase (delta winding) voltages are shown between -VBATT and +VBATT
  - CPWM/SVM waveforms are centered at VBATT/2
  - DPWM modes show top/bottom clamping as in real inverters
- Real-time modulation amplitude control (0–100%)
- Config save/load (JSON) for sharing or reproducing simulations
- Selectable LPF cutoff frequency (or auto 3× electrical frequency)
- Oscilloscope-style scrolling (pause/step/hold)
- Export waveform/FFT to CSV; export plots to PNG
- Parameter sweep mode to plot THD vs speed or PWM frequency
- Dual THD display:
  - THD for line voltage A
  - THD for phase voltage AB
- Simulation speed quantization:
  - Uses an integer number of PWM pulses per electrical cycle
  - Displays requested speed, realizable speed, and speed deviation
- PWM timing realism:
  - Pulse alignment modes: left-aligned, right-aligned, center-aligned
  - Configurable dead time insertion without changing the PWM period
  - Dead-time open-leg diode conduction model using current direction
  - User-settable diode forward voltage (default 0.6 V)
  - Synthetic current phase control (default 30 deg, range -45 deg to +45 deg)
- FFT and metrics computed over 10 electrical cycles by default
- Concise voltage metrics display:
  - One line-voltage metric (A) and one phase-voltage metric (AB)
  - Mean, RMS, min, and max for each
- Common-mode voltage (CMV) plot: live CMV waveform panel with mean/RMS/min/max/peak-to-peak metrics
- DC bus current ripple plot: normalised ripple waveform with peak-to-peak and RMS metrics
- Comparison mode: save a simulation snapshot as reference; grey dashed overlays on waveform, FFT, and duty plots; info box shows ΔTHD, ΔCMV pp, and ΔDC bus pp
- Accessibility: keyboard navigation, screen-reader-friendly labels
- Common mode voltage (CMV) panel:
  - Plots CMV = (Va + Vb + Vc) / 3 on a dedicated scrollable panel
  - Displays mean, RMS, min, max, and peak-to-peak statistics in the info box
  - Show/hide checkbox; colour-coded purple
- DC bus current ripple panel:
  - Plots the normalised DC bus current ripple I_dc_norm = Da·sin(ωt+φ) + Db·sin(ωt−2π/3+φ) + Dc·sin(ωt+2π/3+φ)
  - Reports min, max, RMS, and peak-to-peak in the info box
  - Show/hide checkbox; colour-coded red
- Comparison mode:
  - "Save Reference" button freezes a simulation snapshot as a grey dashed overlay on the waveform, FFT, and duty cycle plots
  - "Clear Reference" button removes all overlays
  - Info box shows ΔTHD, ΔCMV peak-to-peak, and ΔDC bus peak-to-peak against the saved reference

## Running

```sh
python main.py
```

## Development

Install dependencies:

```sh
python -m pip install -r requirements.txt
```

Build a Windows executable (GUI, one-file):

```sh
python -m PyInstaller --name svm-analyst --onefile --windowed main.py
```

The generated executable will be available in `dist/svm-analyst.exe`.

Run tests:

```sh
pytest
```

## Project Structure

- `svm_shaper/` - core simulation modules and GUI
- `docs/` - documentation (Sphinx)
- `USER_MANUAL.md` - end-user operating manual for the packaged application
- `docs/SVM-Analyst-User-Manual-1.1.2.pdf` - PDF export of the user manual
- `tests/` - unit and integration tests

## License

See `LICENSE`.
