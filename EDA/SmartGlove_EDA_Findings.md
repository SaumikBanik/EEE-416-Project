# Scientific interpretation

Verdict: **READY FOR PREPROCESSING, WITH SPECIFIC TRIALS/ISSUES TO REVIEW**

1. **Structurally complete? DIRECT EVIDENCE:** 500 accepted trial records, 25000 samples, 100 gesture files; every participant × gesture cell contains [5] trials. The five-repetition protocol is met when the audit passes. Ten people, not 500 independent people, underpin generalization.

2. **Corrupt/incomplete accepted trials? DIRECT EVIDENCE:** 0 missing cells, 0 infinite cells, 0 within-trial duplicate rows, and 0 wrong-length trials. Accepted CSV/log keys are one-to-one. NO EVIDENCE: this cannot independently verify a participant performed the stated gesture.

3. **Sampling timing? DIRECT EVIDENCE:** exact 0–1960 ms grids at 40 ms steps; logged min/max intervals 40/40 ms, zero-missed status=True, maximum lateness 87 microseconds. INDIRECT EVIDENCE: these records support the collection schedule. NO EVIDENCE that the CSV grid alone establishes hardware sampling timestamps or sample freshness.

4. **Five flex channels consistent? DIRECT EVIDENCE:** raw limits and per-finger spans are tabulated. Signed spans range -1144.22 to 66.06 ADC. Participant main-effect fractions range 0.3%–9.0%; participant×gesture fractions range 1.5%–17.1%. Different participant/finger response patterns remain important. Thumb reference warnings: [{'participant': 'P07', 'session': 'S01', 'signed_span': 66.05600000000004}]; opposite direction is not by itself proof of hardware polarity reversal.

5. **Saturated, stuck or noisy? DIRECT EVIDENCE:** 0 exact rail samples, 0 out-of-bounds readings; see constant-channel counts in flex_sensor_summary.csv. Largest within-trial finger range is 656 ADC at [{'participant': 'P04', 'gesture_id': 'G05', 'trial_id': 2}]. INDIRECT EVIDENCE: unusually large steps/ranges merit review; no single cause is established. A flat static trace alone does not prove a stuck sensor.

6. **Sufficiently stationary? DIRECT EVIDENCE:** gyro peak median/p95/max=8.155/18.163/37.829 deg/s; quaternion pairwise diameter median/p95/max=1.103/2.805/5.537 degrees. INDIRECT EVIDENCE: many holds show limited quaternion motion. NO EVIDENCE: a validated universal acceptable-motion boundary for this classifier; review joint flex/gyro/quaternion traces. Logged gyro peaks differ from recomputed raw peaks by up to 14.538 deg/s; even bias correction leaves discrepancies. Inspect firmware calculation and sample rate before treating these as the same metric.

7. **Quaternion validity? DIRECT EVIDENCE:** norms 0.99999986–1.00000017, error mean 4.7543672e-08, p95 1.0667456e-07, max 1.7045035e-07. Angular calculations normalize nonzero quaternions and respect sign equivalence. NO EVIDENCE: normalization guarantees orientation accuracy.

8. **Orientation drift? DIRECT EVIDENCE:** same-session neutral-return angles span 3.386–143.799 degrees over all checkpoint/final files. Maximum absolute logged-vs-recomputed difference is 0.0005 degrees. INDIRECT EVIDENCE: large deviations undermine assuming a common absolute orientation across collection. NO EVIDENCE: these deviations are purely yaw drift, or entirely electronic rather than neutral-pose repositioning.

9. **Restarts and segmentation? DIRECT EVIDENCE:** accepted log entries match references, boot and epoch; 16 named sessions include unused calibrations. Inspect session_summary.csv. In this archive P05 final drift belongs to a session without accepted trials; P01 has extra R06 and no final-drift/completion event. INDIRECT EVIDENCE: preserved segmentation makes session-aware analysis possible, not proof that all post-restart features are comparable.

10. **Inter-participant variation? DIRECT EVIDENCE:** inspect per-finger participant, gesture and interaction variance fractions and the same-feature PCA colored two ways. These are descriptive decompositions of this balanced sample. Temporal associations after within-session/gesture demeaning vary: {'collection_sequence': {'median': -0.057, 'min': -0.48, 'max': 0.53}, 'elapsed_session_minutes': {'median': -0.038, 'min': -0.476, 'max': 0.528}, 'round': {'median': -0.034, 'min': -0.479, 'max': 0.534}, 'temp_c': {'median': -0.042, 'min': -0.46, 'max': 0.345}}. No uniform causal temperature effect is established. NO EVIDENCE: population representativeness or unseen-person accuracy.

11. **Will calibration help? DIRECT EVIDENCE:** normalization_variation_comparison.csv and normalization_silhouette_comparison.csv compare the same eligible trials; 500/500 qualify for the exploratory span rule. DIRECT EVIDENCE: gesture silhouette in five-flex space is 0.368 raw, 0.342 neutral-subtracted and 0.159 span-scaled. These results do not support assuming that calibration improves class geometry; inspect the thumb references first. INDIRECT EVIDENCE: calibration may introduce reference-posture or denominator problems. NO EVIDENCE: better held-out performance; no model was trained. Calibration after a recording cannot serve as a prospective baseline without an explicit deployment protocol.

12. **Potentially difficult gesture pairs? DIRECT EVIDENCE:** the lowest raw-flex centroid/radius ratio is G05 (This Way) versus G08 (Victory), ratio 0.229. Full pair rankings are saved. INDIRECT EVIDENCE: these pairs may deserve feature/trace investigation. NO EVIDENCE: predicted confusion rates.

13. **Which exact trials need inspection? DIRECT EVIDENCE:** 64 distinct accepted trials receive at least one exploratory flag. qc_flagged_trials.csv is the complete exact list; qc_flags_long.csv contains each metric and numerical threshold. priority_manual_review.csv and the six trace plots start with largest flex ranges. The review count is not a count of bad trials.

14. **Exclude anything before preprocessing?** Structural evidence does not justify blanket deletion of accepted trials when the hard audit passes. Keep the nine discarded recordings and failed attempts out of accepted data by their existing status. Confirm any accepted-trial exclusion by documented trace/protocol evidence, with sensitivity analysis. NO EVIDENCE: an empirical outlier by itself is invalid.

15. **Document versus correct?** Correct ambiguous metadata only after checking collection history; retain originals and a change log. Complete gesture-definition placeholders. Document retries, calibration rejection events, P01 checkpoint/completion irregularities and calibration-only sessions. Do not fabricate final checks or merge orientation chains across references.

16. **Next preprocessing experiments:** preserve one row per trial and participant grouping; compare robust flex medians alone versus variability summaries, neutral-subtracted and signed-span flex, and flex plus gravity/acceleration features. Audit absolute quaternion/Euler features before use because long-session deviation is large. Investigate isolated flex steps before choosing smoothing or clipping. Fit scaling/feature selection on training participants only. Use a participant-held-out outer loop and participant-grouped inner loop for tuning; if unseen-person calibration will be available, simulate its exact timing and labeled-reference availability. Report per-participant variability, not just pooled performance. No KNN or final classifier is trained here.

**READY FOR PREPROCESSING, WITH SPECIFIC TRIALS/ISSUES TO REVIEW**
