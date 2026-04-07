# SVM Analyst – Feature Roadmap

This roadmap lists the agreed future features in priority order.
Features marked **Done** are already shipped; features marked **In Progress** are
being developed in the current sprint.

---

## Batch 1 — High value, easiest to implement

| #   | Feature                                                                                                                 | Status  | Target version |
| --- | ----------------------------------------------------------------------------------------------------------------------- | ------- | -------------- |
| 1   | **Common Mode Voltage (CMV) plot** — `(Va+Vb+Vc)/3` panel, key EMC metric                                               | ✅ Done | v1.2.4         |
| 2   | **DC bus current ripple** — `Da*Ia + Db*Ib + Dc*Ic` (normalised) per-PWM-period                                         | ✅ Done | v1.2.4         |
| 3   | **Side-by-side comparison mode** — "Save Reference" snapshot, dashed overlay on waveform / duty / FFT, ΔTHD in info box | ✅ Done | v1.2.4         |
| 4   | **Duty cycle staircase in PDF report** — matplotlib `step(where="mid")` so the PDF matches the GUI ZOH display          | ✅ Done | v1.2.4         |

---

## Batch 2 — Medium effort

| #   | Feature                                                                                                                                             | Status  | Target version |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | -------------- |
| 5   | **dq-frame phasor diagram** — voltage and current vectors in d-q plane via pyqtgraph polar plot or dialogue                                         | Planned | TBD            |
| 6   | **Overmodulation region (MI > 1)** — extend core duty cycle computation beyond unity modulation index, visualise saturation and six-step transition | Planned | TBD            |
| 7   | **Interleaved dual-inverter mode** — two sets of phase-shifted carriers, DC bus ripple cancellation analysis                                        | Planned | TBD            |

---

## Batch 3 — Differentiators

| #   | Feature                                                                                                                                          | Status  | Target version |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | -------------- |
| 8   | **Audible PWM preview** — stream one cycle of the PWM output through `sounddevice` so acoustic-noise differences between strategies can be heard | Planned | TBD            |
| 9   | **Switching loss + junction temperature estimation** — MOSFET/IGBT Eon/Eoff model, per-switch loss, junction temperature rise estimator          | Planned | TBD            |
| 10  | **Custom modulation via Python script** — allow users to define their own `compute_duty(theta, mi)` function as a plugin loaded at runtime       | Planned | TBD            |

---

## Out of scope

- Full FOC loop visualiser → carried by the dedicated **SPINOTOR** project.

---

_Last updated: v1.2.4_
