# SVM Shaper

Educational simulator for PWM and space-vector modulations focused on PMSM drives.

## Features

- Simulates multiple modulation techniques from the provided thesis PDF:
  - THIPWM 1/6 and 1/4
  - Space vector modulation (SVM)
  - Discontinuous PWM (DPWM): 120°, 60° variants, 30° (DPWM3)
- Interactive GUI with real-time waveform and FFT visualization
- Display options:
  - Phase voltages vs line voltages
  - PWM waveform vs filtered (fundamental) waveform
  - Optional switching-edge markers for PWM transitions
  - SVM sector highlighting and active sector indicator
- Realistic voltage scaling:
  - Phase outputs are shown between 0 V and battery voltage (VBATT)
  - CPWM/SVM waveforms are centered at VBATT/2
  - DPWM modes show top/bottom clamping as in real inverters
- Real-time modulation amplitude control (0–100%)
- Config save/load (JSON) for sharing or reproducing simulations
- Selectable LPF cutoff frequency (or auto 3× electrical frequency)
- Oscilloscope-style scrolling (pause/step/hold)
- Export waveform/FFT to CSV; export plots to PNG
- Parameter sweep mode to plot THD vs speed or PWM frequency
- Real-time THD computation (on filtered waveform) and top harmonics display
- Accessibility: keyboard navigation, screen-reader-friendly labels

## Running

```sh
python main.py
```

## Development

Install dependencies:

```sh
python -m pip install -r requirements.txt
```

Run tests:

```sh
pytest
```

## Project Structure

- `svm_shaper/` - core simulation modules and GUI
- `docs/` - documentation (Sphinx)
- `tests/` - unit and integration tests

## License

See `LICENSE`.
