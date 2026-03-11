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

Packaging example:

.. code-block:: sh

   python -m PyInstaller --name svm-analyst --onefile --windowed main.py
