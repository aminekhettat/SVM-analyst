Welcome to SVM Analyst's documentation!
=====================================

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
- THD computed on the filtered output with top-harmonics listing
- Optional switching-edge markers for PWM gate events
- SVM hexagon view with active sector highlighting
- Save/load simulation configuration (JSON)
- Export waveform/FFT to CSV and plots to PNG
