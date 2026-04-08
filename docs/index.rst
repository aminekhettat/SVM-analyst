Welcome to SVM Analyst's documentation!
==================================================

.. toctree::
   :maxdepth: 2

   api


Project overview
----------------

SVM Analyst is an educational simulator for PWM modulation techniques used in
three-phase inverters. It provides an interactive GUI to visualize PWM waveforms
and harmonic spectra, and includes a small Python API for batch analysis.

Key features:

- Multiple modulation methods (THIPWM, SVM, DPWM variants)
- Real-time waveform + FFT plotting with oscilloscope-style scrolling
- Filtered waveform view with selectable LPF cutoff
- Dual THD computation and display:

   - THD for line voltage A (terminal-to-ground)
   - THD for phase voltage AB (terminal-to-terminal)

- Integer PWM-pulse quantization per electrical cycle, with requested vs real speed reporting
- PWM alignment modes (left, right, center) for MCU-like timer behavior
- Configurable dead time insertion without altering PWM period
- Dead-time open-leg diode conduction model with configurable diode forward voltage
- Synthetic current phase control (default 30 deg, adjustable from -45 deg to +45 deg)
- FFT and metrics computed over 10 electrical cycles by default
- Concise metrics display for one line voltage (A) and one phase voltage (AB): mean, RMS, min, max
- Optional switching-edge markers for PWM gate events
- SVM hexagon view with active sector highlighting
- Save/load simulation configuration (JSON)
- Export waveform/FFT to CSV and plots to PNG
- Build a standalone Windows executable with PyInstaller
- Common mode voltage (CMV) panel:

   - CMV = (Va + Vb + Vc) / 3, scrollable, show/hide checkbox
   - Statistics: mean, RMS, min, max, peak-to-peak in the info box

- DC bus current ripple panel:

   - Normalised ripple I_dc_norm from three-phase duty cycles, scrollable, show/hide checkbox
   - Statistics: min, max, RMS, peak-to-peak

- Comparison mode:

   - Save/clear a reference simulation snapshot
   - Grey dashed overlays on waveform, FFT, and duty cycle plots
   - Delta metrics in the info box: ΔTHD, ΔCMV pp, ΔDC bus pp

- dq-frame phasor diagram:

   - Clarke αβ trajectory, Park dq phasors, electrical and mechanical angle sawtooth
   - αβ/dq metric footer: Vα, Vβ, Vd, Vq, |Vαβ| and |Vdq| RMS/peak/mean
   - Real-time pyqtgraph PlotWidgets for flicker-free refresh

- Overmodulation region (MI > 1):

   - Modulation index (MI) spinbox, range 0.10 – 1.50, default 1.0
   - Pre-comparison reference scaling: MI > 1 clamps duty cycles to 0 % or 100 %
   - Linear boundary: MI = 1.0 for sinusoidal/THIPWM, MI ≈ 1.15 for SVM/DPWM
   - Inline "⚠ OM X%" warning label and saturation % in the info panel
   - Supports full progression from linear region through deep overmodulation to
     six-step operation

Packaging example:

.. code-block:: sh

   python -m PyInstaller --name svm-analyst --onefile --windowed main.py
