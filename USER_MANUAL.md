# SVM Analyst User Manual

Version 1.0.0

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

## 6. Reading the Plots

### Waveform Plot

The upper plot shows either PWM terminal voltages or phase-to-phase voltages depending on the selected display mode.

- Line voltages are terminal-to-ground and range from 0 to the DC bus voltage.
- Phase voltages are terminal-to-terminal and range from negative to positive DC bus voltage.

### FFT Plot

The spectrum highlights the fundamental component and switching-related harmonics. The reported THD values are computed from the filtered analysis signals.

## 7. Key Metrics

The information panel summarizes the most relevant values:

- Line voltage A mean, RMS, minimum, and maximum
- Phase voltage AB mean, RMS, minimum, and maximum
- THD for line voltage A
- THD for phase voltage AB
- Requested speed, realizable speed, and deviation
- Average phase PWM pulses per electrical cycle

## 8. Display Options

You can switch between several display configurations:

- PWM waveform versus filtered waveform
- Line-voltage view versus phase-voltage view
- Switching-edge markers on or off
- SVM hexagon with active-sector indication

These options are useful when comparing common-mode behavior, clamping behavior, and harmonic tradeoffs across modulation families.

## 9. Exports

### CSV Export

Use CSV export when you need numerical waveform or FFT data for external processing.

### PNG Export

Use PNG export to save the current visualization for reports or presentations.

### PDF Report Export

The PDF report includes the current configuration, waveform and FFT plots, summary metrics, and explanatory notes.

## 10. Configuration Files

You can save the current configuration to JSON and reload it later. This is useful for reproducible comparisons between modulation methods.

## 11. Parameter Sweep Mode

Sweep mode helps compare how THD changes as speed or PWM frequency varies. Use it to identify operating regions where a given modulation method performs best.

## 12. Troubleshooting

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

## 13. Recommended First Comparison

For a quick introduction:

1. Start with Sinusoidal PWM.
2. Note the waveform shape, THD, and pulse count.
3. Switch to SVM with the same parameters.
4. Compare the FFT and voltage metrics.
5. Switch to a DPWM mode and observe the clamped segments and reduced switching activity.

## 14. Support Material

The repository also includes developer-oriented API documentation in the `docs` folder and generated HTML documentation in `docs/_build`.
