# SVM Analyst User Manual

Version 1.1.2

## What's New

### Version 1.1.2

- Fixed critical bug: scipy was incorrectly excluded from the PyInstaller bundle,
  causing the executable to crash on launch with "no module named scipy".
- Removed PyQt6 from the build environment; PySide6 is now the sole Qt binding,
  eliminating the PyInstaller Qt-bindings collision error.

### Version 1.1.1

- Application icon embedded in the executable and window for professional visual identity.
- Automated distribution build script producing a ready-to-distribute ZIP package.
- Freshly regenerated user manual and API documentation aligned with the executable.
- Example files included in the distribution: default configuration and sample waveform.

### Version 1.1.0

- Unified name "SVM Analyst" applied to all window titles, file prefixes, and menus.
- Full accessibility support: every control has an accessible name and description
  compatible with screen readers such as NVDA and JAWS.
- Extended test suite covering 191 tests: modulations, analysis, visualization,
  sweep, IO, accessibility, and GUI behavior.
- GitHub Actions CI pipeline for automated quality gates on each push.

### Version 1.0.2

- Migrated GUI from PyQt6 to PySide6.
- Added pyqtgraph oscilloscope: real-time scrolling waveform with pause, step, and zoom.

## 1. Purpose

SVM Analyst is a desktop tool for exploring PWM strategies used in three-phase inverters. It lets you compare sinusoidal PWM, third-harmonic injection, space-vector modulation, and discontinuous PWM variants while inspecting time-domain waveforms, FFT spectra, THD, and practical switching metrics.

## 2. System Requirements

- Windows 10 or Windows 11
- A 64-bit CPU
- A display resolution of at least 1280 x 720

## 3. Starting the Application

### Executable build

If you received the packaged release, start the application by double-clicking `svm-analyst.exe`.

### Python source run

If you are running from source:

```sh
python main.py
```

## 4. Main Workflow

1. Choose the modulation method.
2. Set the machine and inverter parameters.
3. Select whether you want to inspect line voltages or phase voltages.
4. Review the waveform plot, FFT, THD values, and summary metrics.
5. Export CSV, PNG, or PDF reports if needed.

## 5. Main Parameters

### Modulation

Select the PWM strategy to simulate:

- Sinusoidal PWM
- THIPWM 1/6
- THIPWM 1/4
- SVM
- DPWM 120 max and min
- DPWM 60 variants
- DPWM 30 variant

### PWM Frequency

Defines the switching frequency in hertz. Higher values reduce visible ripple in the filtered waveform but increase switching events.

### PWM Alignment

Defines how pulses are aligned in each PWM period:

- Left-aligned
- Right-aligned
- Center-aligned

This mirrors common MCU timer modes and changes switching instant placement.

### Dead Time

Dead time inserts a short non-switching interval around commutations to emulate real gate-driver timing constraints.

In version 1.0.2 and later, dead time is modeled at inverter leg level with two key behaviors:

- The PWM period remains constant (for example, 20 kHz remains 50 us period).
- During dead time, the open-leg voltage is set by diode conduction and current direction.

Increasing dead time affects switching behavior and can impact THD.

### Diode Forward Voltage

Defines the diode forward drop used during dead-time conduction. Default is 0.6 V.

Dead-time open-state voltage follows current direction:

- Positive current: approximately -Vf
- Negative current: approximately Vbatt + Vf

This parameter helps emulate practical bridge-leg behavior during non-overlap intervals.

### Current Phase

Defines the phase of the synthetic current signal used for dead-time polarity selection.

- Default: 30 degrees
- Range: -45 degrees to +45 degrees

The synthetic current uses the same electrical frequency as the voltage reference and is phase-shifted by this parameter.

### Speed

Defines the requested mechanical speed in RPM. The simulator quantizes operation to an integer number of PWM pulses per electrical cycle and reports the realizable speed and deviation.

### Pole Pairs

Defines the conversion between mechanical speed and electrical frequency.

### Battery Voltage

Defines the DC bus voltage used by the inverter model.

### Amplitude

Sets the requested modulation depth as a percentage.

### Low-Pass Filter Cutoff

Controls the filtered waveform display used for fundamental analysis. Set to zero to use the automatic default.

## 6. Input Constraints and Validation

The numeric parameter controls are constrained in the GUI to avoid invalid combinations.

- Numeric entries are controlled by spin boxes and constrained to valid ranges.
- Dead time maximum is auto-limited from the PWM period (kept below half-period).
- Current phase is limited to -45 to +45 degrees.

When dependent settings change (for example PWM frequency), limits are updated automatically.

## 7. Reading the Plots

### Waveform Plot

The upper plot shows either PWM terminal voltages or phase-to-phase voltages depending on the selected display mode.

- Line voltages are terminal-to-ground and range from 0 to the DC bus voltage.
- Phase voltages are terminal-to-terminal and range from negative to positive DC bus voltage.

### FFT Plot

The spectrum highlights the fundamental component and switching-related harmonics. The reported THD values are computed from the filtered analysis signals.

## 8. Key Metrics

The information panel summarizes the most relevant values:

- Line voltage A mean, RMS, minimum, and maximum
- Phase voltage AB mean, RMS, minimum, and maximum
- THD for line voltage A
- THD for phase voltage AB
- Requested speed, realizable speed, and deviation
- Average phase PWM pulses per electrical cycle

## 9. Display Options

You can switch between several display configurations:

- PWM waveform versus filtered waveform
- Line-voltage view versus phase-voltage view
- Switching-edge markers on or off
- SVM hexagon with active-sector indication

These options are useful when comparing common-mode behavior, clamping behavior, and harmonic tradeoffs across modulation families.

## 10. Exports

### CSV Export

Use CSV export when you need numerical waveform or FFT data for external processing.

### PNG Export

Use PNG export to save the current visualization for reports or presentations.

### PDF Report Export

The PDF report includes the current configuration, waveform and FFT plots, summary metrics, and explanatory notes.

The exported report also includes PWM alignment, dead time, diode forward voltage, and current phase settings.

## 11. Configuration Files

You can save the current configuration to JSON and reload it later. This is useful for reproducible comparisons between modulation methods.

## 12. Parameter Sweep Mode

Sweep mode helps compare how THD changes as speed or PWM frequency varies. Use it to identify operating regions where a given modulation method performs best.

## 13. Troubleshooting

### The executable does not start

- Verify that the release files were extracted completely.
- Run the executable from a local folder with write permission.
- If Windows SmartScreen appears, review the publisher details and allow execution if appropriate in your environment.

### The plots look empty or flat

- Check that amplitude is greater than zero.
- Verify that battery voltage and speed values are valid.
- Make sure the selected waveform mode matches the signal type you expect.

### The realizable speed differs from the requested speed

This is expected. The simulator uses an integer number of PWM pulses per electrical cycle, so the exact requested speed may not always be achievable for the selected switching frequency and pole-pair count.

### DPWM waveforms look clamped

This is expected. DPWM intentionally clamps one phase to the top or bottom rail over part of the electrical cycle to reduce switching losses.

### Dead-time plateaus go slightly below 0 V or above Vbatt

This is expected with the diode conduction model. The dead-time open-leg voltage includes the configured diode drop and depends on the synthetic current direction.

## 14. Recommended First Comparison

For a quick introduction:

1. Start with Sinusoidal PWM.
2. Note the waveform shape, THD, and pulse count.
3. Switch to SVM with the same parameters.
4. Compare the FFT and voltage metrics.
5. Switch to a DPWM mode and observe the clamped segments and reduced switching activity.

## 15. Support Material

The repository also includes developer-oriented API documentation in the `docs` folder and generated HTML documentation in `docs/_build`.
