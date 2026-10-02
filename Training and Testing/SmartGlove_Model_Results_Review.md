# SmartGlove model results: independent review

## Decision

**Proceed with logistic regression as a provisional deployment candidate. The supplied evidence does not show a material implementation error that explains its lead. However, the results do not establish that logistic regression is reliably superior to the other leading models. Its advantage over the MLP is one trial out of 500, concentrated in one participant.**

The strongest empirical finding is that adding gravity-direction information substantially improves recognition over flex-only inputs. The weakest part of the evidence is generalization beyond ten people and the controlled acquisition protocol. Resolve reproducibility documentation, investigate the difficult participants and gestures, then test a frozen pipeline on new people and sessions. Do not tune until a preferred model wins.

## What was examined and independently verified

Inputs: `SmartGlove_Dataset(3).zip`, `SmartGlove_EDA_Kaggle.ipynb`, `EDA_results(3).zip`, `SmartGlove_Kaggle_Fixed(2).ipynb`, and `results(1).zip`.

The review covered the EDA workflow and outputs, training feature construction, calibration joins, all declared model grids, nested selection logic, preprocessing placement, saved split manifests, prediction tables, metrics, warnings, final-model metadata and exported inference source. Original input files were not changed.

Verification performed:

- All three ZIP archives passed integrity checks. The current dataset and EDA archive hashes equal those of the previously reviewed dataset and EDA. These are the same data and EDA outputs despite their new filenames.
- Independently recomputed the dataset-directory hash. It exactly matches the hash in the training results.
- Reconstructed all six training feature tables from the raw trial recordings and exported session references. Largest absolute discrepancy was approximately 2.27 × 10⁻¹³, attributable to floating-point representation.
- Rechecked all 500 accepted trials: 50 samples per trial, finite numeric values and timestamps 0, 40, …, 1960 ms. Each of the six feature tables has zero duplicate complete predictor vectors.
- Verified the 11 split manifests, comprising 90 inner folds for outer evaluation and 10 folds for final tuning. Participant groups are disjoint and each outer test person is absent from that fold's inner training and validation groups.
- Recomputed accuracy and macro-F1 for all 60 classifier-level outer rows and all 360 fixed-feature-set/classifier outer rows. Maximum difference from the saved metrics was approximately 1.11 × 10⁻¹⁶.
- Checked that all saved overall, classifier-level and feature-set/classifier selections follow the recorded inner-CV score and tie-breaking rules. No selection mismatches were found.
- Independently fitted the 60 classifier-level selected outer models using the recorded settings. **All 3,000 predicted labels matched the saved predictions exactly.**
- Independently repeated ten-fold participant-held-out scoring for the final selected logistic-regression configuration and its closest final-search SVM competitor. Their mean accuracy and macro-F1 match the saved values.

Scope of the refit verification: the original run used scikit-learn 1.6.1 and NumPy 2.0.2; the audit environment used scikit-learn 1.8.0 and NumPy 2.3.5. The removed `multi_class='deprecated'` constructor argument was omitted in the audit, retaining multinomial logistic regression. Random Forest used one worker rather than all workers. Exact prediction agreement despite these differences supports the result, but is not a byte-for-byte recreation of the original environment. The full 63,000-inner-fit search was not repeated. Saved serialized model/checkpoint objects were not needed to establish the prediction agreement; the final bundle's binary contents were not independently certified.

Provenance:

| Item | SHA-256 |
|---|---|
| Dataset ZIP | `b3a601cd88bd0ab8cefb8cc1f297fa5050cc61f2e1b249e48716298fec4cb30e` |
| EDA ZIP | `e0387b7948e8dc60c65cde93d250cb2affe2283527755566f114a60b3e16fb9c` |
| Results ZIP | `5cb21292097e8ba616c1be38c597d7858c40d3eb20729610cd3e7b25491e590f` |
| Dataset directory fingerprint used by training | `4217ffd727eb260a0867eb547b16111ddecf40d4c8c974bdab890755b6a8649b` |

The directory hash differs from the ZIP hash because the two hash different byte sequences. This is expected. The reviewed run ID is `ed80fca0a01db881dcb96e19a8bb982a0e9424c57558e78a16012c5ae1635e89`.

## What the model comparison actually says

From `outer_classifier_summary.csv` and independently checked `outer_classifier_predictions.csv`:

| Model family | Correct / 500 | Accuracy | Mean participant macro-F1 | Participant accuracy SD |
|---|---:|---:|---:|---:|
| Logistic regression | 493 | 98.6% | 0.9844 | 3.13 percentage points |
| Small MLP | 492 | 98.4% | 0.9824 | 3.10 percentage points |
| KNN | 490 | 98.0% | 0.9784 | 3.13 percentage points |
| LDA | 490 | 98.0% | 0.9785 | 3.65 percentage points |
| SVM | 490 | 98.0% | 0.9770 | 3.77 percentage points |
| Random Forest | 480 | 96.0% | 0.9502 | 8.89 percentage points |

For each family, the feature set and parameters were selected separately inside each outer fold. Thus, “logistic regression 98.6%” describes the logistic-regression family with inner selection; it does not describe a single fixed F5 configuration used in every fold.

The MLP and logistic regression have identical correctness on 499 trials. On the remaining trial, P10–G08–T2, logistic regression is correct and the MLP predicts G05. Their participant accuracies tie on nine people; logistic regression wins on one. This is insufficient evidence to declare a dependable population-level superiority.

Against SVM, logistic regression is correct on five trials SVM misses, while SVM is correct on two trials logistic regression misses. Their participant accuracies tie on eight people; each wins on one of the other two. The 0.6-percentage-point aggregate difference is small relative to the limited participant evidence.

The reported SD is variation across participants, not a confidence interval. The 500 repeated trials are clustered within ten people. Treating all 500 as independent observations in a simple significance test would overstate the precision. Even the ten cross-validation folds share much of their training data, so a naive independent-fold test needs caution.

## Keep three performance quantities separate

| Quantity | Verified result | Meaning |
|---|---:|---|
| Overall nested model/feature/parameter selection procedure | 489/500, **97.8%**, mean participant macro-F1 0.9749 | The outer estimate for the procedure that chooses among all six families and all six feature sets |
| Logistic-regression family with feature/parameter selection | 493/500, **98.6%**, mean participant macro-F1 0.9844 | A family-specific nested result; declaring it the winner after comparing families adds a selection step |
| Final all-participant tuning winner | **98.8% tuning accuracy**, mean participant macro-F1 **0.98727994** | Development-CV score used to select the deployment configuration, not a fresh test result |

The overall selection chose logistic regression in five outer folds, SVM in three and MLP in two. Its 97.8% accuracy is lower because inner selection sometimes chooses a configuration that does not perform best on the held-out person. That is an ordinary consequence of model-selection uncertainty, not an averaging bug.

The final exported configuration is **F5_NEUTRAL_FLEX_STATIC_IMU + StandardScaler + logistic regression, C=10**, refitted on all 500 trials. Its final tuning accuracy ties the F3 linear SVM with C=0.1 at 98.8%. Logistic regression wins by the declared mean participant macro-F1 criterion: 0.98727994 versus 0.98655789, a difference of about 0.000722. The models can have equal accuracy and different macro-F1 because their errors affect classes differently.

There is no new independent test score for the final all-participant model. Nested validation evaluates the selection procedure; a future untouched cohort evaluates the frozen exported model.

## The main reason the results are plausible: informative features

The six representations contain:

| Feature set | Predictors |
|---|---|
| F1 | Five raw flex medians |
| F2 | Five session-neutral-centered flex medians |
| F3 | F1 plus three relative gravity components |
| F4 | F2 plus three relative gravity components |
| F5 | F4 plus acceleration-magnitude median/SD and bias-corrected gyro-magnitude median/p95 |
| F6 | F2 plus five flex IQRs |

Relative gravity here means `u − u0`: the normalized componentwise-median trial acceleration vector minus the normalized session-neutral acceleration vector. It is a three-component descriptor of gravity-direction change. It is not a full 3D rotation into a common world frame and cannot recover heading about gravity.

From the fixed-feature-set nested comparisons:

| Representation | Logistic regression | MLP | SVM | KNN | LDA | Random Forest |
|---|---:|---:|---:|---:|---:|---:|
| F1 raw flex | 87.6% | 88.2% | 88.8% | 87.0% | 88.6% | 89.2% |
| F2 centered flex | 89.6% | 88.6% | 87.4% | 90.0% | 89.0% | 89.2% |
| F3 raw flex + gravity | 98.0% | 98.4% | 98.0% | 97.6% | 97.6% | 96.0% |
| F4 centered flex + gravity | 98.2% | 97.6% | 97.6% | 96.8% | 98.4% | 95.8% |
| F5 centered flex + gravity + motion statistics | 98.2% | 98.6% | 97.2% | 95.0% | 97.4% | 96.4% |
| F6 centered flex + IQR | 89.8% | 88.6% | 90.2% | 84.0% | 89.0% | 88.8% |

Logistic regression does not win every row. Random Forest is best in F1, KNN in F2, MLP in F3 and F5, LDA in F4, and SVM in F6. The family-level headline includes the behavior of the inner feature-selection procedure, not just the classifier's mathematical form.

For logistic regression, adding gravity to raw flex improves accuracy by **10.4 percentage points**, from 438/500 to 490/500. Flex-only reciprocal confusions between G05 “This Way” and G08 “Victory” total **41 trials**. With F3 these confusions fall to **7**. This is direct evidence that the accelerometer-derived feature block supplies information missing from the flex-only representation under this protocol.

The result does not prove that every motion statistic in F5 is useful. F4 and F5 logistic regression both score 98.2% in the fixed-representation nested comparison. Nor does it establish individual finger importance. Selection-frequency plots count selection of whole feature blocks and are not independent per-feature importance estimates.

## Why a linear classifier can match or beat nonlinear models

Logistic regression is a regularized multiclass classifier, despite the word “regression.” After preprocessing it learns one weighted score per gesture, then predicts the class with the largest score. Pairwise decision boundaries are linear in the constructed features. The implementation's default L2 penalty controls coefficient size [1].

These inputs are already informative summaries: five stable finger-position estimates, gravity direction and a few motion statistics. The model is not asked to discover hand geometry from images or raw unsegmented sequences. Median aggregation also reduces the effect of isolated sample fluctuations. With 400 inner-training or 450 outer-training trials and just 8–12 predictors in the best representations, a regularized linear separator can work very well.

The observed results support useful approximate linear separability for this controlled task. They do not prove perfect population separability or that the dataset is scientifically trivial. SVM selects a linear kernel in seven of the ten classifier-level outer folds, and LDA is also strong. These independently reinforce the interpretation that elaborate nonlinear boundaries are not essential for most examples here.

Model-specific explanations, with appropriate limits:

- **SVM:** Linear and RBF kernels were both available. Differences in multiclass construction, loss and regularization can yield slightly different errors from logistic regression. The observed 98.0% versus 98.6% is not evidence that SVM is unsuitable.
- **KNN:** Standardization makes units comparable, but does not make every dimension equally informative. Additional motion features can distort neighbor distances. Its F3 accuracy is 97.6%, while F5 is 95.0%; this is consistent with some added dimensions being unhelpful to this distance metric, but is not proof of a single causal feature.
- **Random Forest:** Axis-aligned splits can behave differently from weighted linear combinations when a new participant's feature relationships shift. Its poorer average is concentrated in two people, documented below. It is not broadly unable to fit the training examples.
- **MLP:** Its result is effectively tied with logistic regression at the resolution of this experiment. There is no evidence of general training failure. Only Adam and one seed were tested; an L-BFGS sensitivity analysis is reasonable for small datasets [2], but no optimizer change is guaranteed to improve unseen-participant results.
- **LDA:** A strong linear baseline, with shrinkage options explicitly included. Its assumptions about class distributions differ from logistic regression's decision-boundary objective. Three extra errors do not establish a substantive inferiority.

## The hard cases matter more than the leaderboard

| Participant | LR | MLP | SVM | KNN | LDA | RF |
|---|---:|---:|---:|---:|---:|---:|
| P01 | 100% | 100% | 100% | 100% | 100% | 100% |
| P02 | 100% | 100% | 100% | 100% | 100% | 100% |
| P03 | 90% | 90% | 90% | 90% | 88% | 74% |
| P04 | 98% | 98% | 98% | 98% | 98% | 100% |
| P05 | 100% | 100% | 100% | 98% | 98% | 100% |
| P06 | 100% | 100% | 100% | 100% | 100% | 100% |
| P07 | 100% | 100% | 100% | 100% | 100% | 100% |
| P08 | 98% | 98% | 100% | 98% | 98% | 100% |
| P09 | 100% | 100% | 100% | 100% | 100% | 100% |
| P10 | 100% | 98% | 92% | 96% | 98% | 86% |

Random Forest makes all 20 errors on P03 (13) and P10 (7). It scores 100% on each of the other eight people. This points to participant-specific generalization difficulty, rather than a universally defective model.

Independent refits give mean training accuracy 99.87% for Random Forest versus 96.0% held-out accuracy. LR and MLP both reach 100% training accuracy, with held-out accuracies 98.6% and 98.4%. These gaps support investigating fit/generalization differences; they do not by themselves establish which physical source causes them.

P03's G05 gravity features differ strongly from those of the same gesture in other participants:

| Feature | G05 range, other nine people | G05 range, P03 |
|---|---:|---:|
| `gravity_rel_x` | 0.2147 to 0.6491 | −0.0586 to 0.0688 |
| `gravity_rel_y` | 0.1988 to 0.6541 | −0.0011 to 0.1260 |

P03's G07 x/y gravity ranges likewise lie below the same-gesture ranges in the other participants. Some P03 flex values are also outside the other participants' same-gesture ranges. This is measured distribution shift. Possible physical causes include different wrist/arm pose, calibration posture, mounting/fit, or genuine execution variation. The files cannot determine which explanation is correct.

All seven LR errors are:

| Participant | True gesture | Trial(s) | Prediction |
|---|---|---|---|
| P03 | G05 This Way | 2, 3, 4, 5 | G08 Victory |
| P03 | G07 Hey You! | 2 | G10 Attention |
| P04 | G08 Victory | 1 | G03 Stop |
| P08 | G02 Excellent | 3 | G03 Stop |

Review these against raw traces, session events and any observation/video records. Do not remove P03 or other difficult examples because they lower accuracy. Correction or exclusion requires independent evidence of mislabeling, corruption or a protocol violation, plus documentation and sensitivity analysis.

## Audit of handling and implementation

| Check | Finding |
|---|---|
| Sample unit | Correct: one vector per full trial, not 25,000 rows treated as independent observations |
| Group separation | Correct in code and saved manifests: all sessions of a participant stay together |
| Inner selection | Correct: mean participant macro-F1; feature set and parameters selected without outer-test scores |
| Scaling | Fitted on each inner-training partition; outer scaler fitted on nine development participants |
| Metadata leakage | Predictor allowlists exclude participant, gesture label, trial ID, session, ref ID, round and position |
| Calibration | Exact participant/session/reference mapping; previous reference used for the corresponding session |
| Held-out calibration | Declared use of a new person's pre-gesture neutral reference, consistent with a calibrated deployment protocol |
| Rejected attempts | Not included as accepted training observations |
| P07 fist anomaly | Span normalization excluded from F1–F6; the problematic fist denominator cannot explain these model scores |
| Absolute orientation | Raw quaternion/Euler angles are not predictors in these six sets |
| Flex aggregation | Median of 50 samples; reconstructed correctly |
| Gyro processing | Counts converted to degrees/second; session bias subtracted axis by axis before magnitude |
| Search coverage | 105 configurations per feature set: LR 3, KNN 28, SVM 20, RF 36, LDA 6, MLP 12 |
| Search completion | 6,300 outer-development candidate rows and 630 final-development candidate rows present |
| Metrics and labels | Independently recomputed correctly; exported predictions reproduced |
| Random Forest scaling | Unnecessary in principle, but standardization is not a credible explanation for this large performance difference; monotonic rescaling preserves ordinary threshold partitions apart from numerical effects |

The saved run records 63,000 inner-model fits, 360 outer-model fits, 600 preprocessing fits and one final refit. The grids are substantive; this is not an untuned logistic-regression comparison against defaults for every competitor.

**Warnings:** the 1,680 logistic-regression warnings during nested evaluation and 181 during final tuning/refit concern deprecated SciPy L-BFGS-B display options. They are not convergence failures. The two MLP convergence warnings both concern F6, alpha=0.001, architecture (16,8), in inner fits. They do not involve the F3/F5 architecture selected in the MLP family comparison. Independently refitted selected MLPs stopped after 386–424 iterations, below the 2,000 limit.

**Notebook execution history needs repair.** The attached training notebook contains saved `NameError` outputs for `trial_meta` and `extract_feature_pool`, and many cells have no execution count. This indicates an incomplete/out-of-order saved notebook state. Those outputs do not document a clean end-to-end execution. They also do not invalidate the separately completed result archive: the features, splits, metrics, selections and predictions in that archive have been cross-checked independently. Save a fresh successfully executed notebook with the matching results and versions. Reuse compatible checkpoints if appropriate; there is no need to rerun tens of thousands of fits merely to remove stale displayed errors.

**Tie-breaking is explicit.** Exact ties prefer fewer predictors, then declared representation/model/parameter order. LR comes first in model order. For the P03 outer fold, 42 candidate configurations tie at the maximum inner score across LR/SVM/RF; the rule helps choose LR there. Other folds need not be ties. The final LR-versus-SVM choice is not an exact tie in macro-F1. This policy should be reported, but is not a scoring error.

## Remaining limitations and small software improvements

1. **EDA informed the candidate design on this same cohort.** Nested validation protects the automated search it encloses. It does not retroactively hide information used to design F1–F6 or revise the pipeline. Describe these as internal development results and confirm the frozen design on fresh participants. EDA-driven design is useful; its evaluation limits should be explicit [3,4].
2. **Only ten participant units.** More repetitions from the same person will not replace diversity of people, glove fit and sessions. Cohort-wide success also does not demonstrate robustness across days, remounting or unrestricted arm positions.
3. **Controlled static windows only.** These results assume the gesture has already been formed and a complete 50-sample window is available. No background/rest/unknown/transition class was evaluated. A live system may confidently assign an arbitrary movement to one of the ten known gestures.
4. **Gravity dependence must fit the intended task.** If hand orientation defines a gesture, its discriminative value is appropriate. If arbitrary arm tilt should be allowed, robustness remains untested. Standardized collection can make models rely on consistent poses that may change in use. The current data cannot distinguish legitimate orientation information from every acquisition-related correlation.
5. **One stochastic seed.** RF and MLP selection stability across seeds was not established. Their performance should be described for the declared seed/search budget, not as the maximum achievable performance of those model families.
6. **Restricted grids.** The LR final C=10 is the largest tested C; selected MLPs use the largest one-layer architecture tested, (32,), and the smallest alpha, 0.0001. These boundaries justify a small predeclared sensitivity analysis if needed. They do not justify an unlimited search on already-inspected outer scores.
7. **Inference accepts more inputs than F1 mathematically needs.** The helper checks all flex/acceleration/gyro/quaternion columns even for flex-only F1, although F1 predictors are only flex medians. For future sensor-removal experiments, validate only inputs required by the chosen representation. This does not affect the current complete dataset.
8. **Diagnostic-only fist data can still stop preprocessing.** The calibration builder requires fist files and rejects zero spans even though no F1–F6 representation uses span normalization. Separate diagnostic failure from the validity of a neutral-only predictor path when generalizing the loader to future data. This did not remove any current trial or cause the observed ranking.
9. **Export parity test is limited.** The notebook checks all six representations on one trial. Extend parity tests to all participants/references when revising the inference implementation. Training uses logged gyro bias; the helper's default estimated bias from neutral samples is not exactly the same estimator. Preserve the declared bias source for F5. The exported source also allows absent timestamps, so the live caller must guarantee the acquisition interval.
10. **No reliable feature-importance ranking from selection frequency.** All five flex features enter together in F1–F6. Selection frequency cannot show which finger is indispensable. Use grouped held-out permutation or sensor-removal experiments with clear evaluation boundaries [5].

## Recommended next steps

### First: preserve and report this result

- Keep the original results, data hashes, package versions and complete candidate search tables.
- Save the clean executed notebook and matching final model metadata. Retain the calibrated 50-sample feature definitions exactly.
- Report all six models, participant variation, the feature-set comparison and the overall 97.8% nested selection estimate. State that LR has the highest observed family-level accuracy, with a one-trial advantage over MLP.
- Do not claim 98.8% as independent final-model test accuracy, and do not call 98.6% proof that logistic regression is universally superior.

### Second: inspect the meaningful failure modes

- Review P03's G05/G07 and relevant neutral references, plus P10's G08. Confirm gesture pose definitions, especially wrist/arm direction and finger differences for This Way versus Victory.
- Inspect the seven LR mistakes listed above. Record whether there is independent evidence of an error in collection; retain ordinary difficult examples.
- Ensure the deployment calibration posture, sensor placement, window timing and gyro-bias definition match training.

### Third: freeze a candidate and collect confirmation data

Use the exported F5 LR model as the current candidate. Before observing new test outcomes, lock preprocessing, model, C, feature order, calibration, and scoring. New participants must not enter training until their confirmation evaluation is complete.

Collect a feasible new cohort of several people; five to ten additional people would be a useful practical expansion, not a statistically guaranteed minimum. Include repeated sessions, glove removal/replacement and the arm poses actually intended for use. Preserve participant/session grouping and record every trial and rejection consistently. If the task includes free-running recognition, collect rest, transitions and unknown gestures as a separate evaluation requirement.

Judge success against a predeclared practical target, including per-participant and difficult-class performance, rather than only the mean accuracy. If alternatives are evaluated on this cohort and then selected, that cohort becomes selection data; retain a further untouched confirmation set for the newly selected pipeline.

### Optional, bounded methodological sensitivity checks

If a research comparison requires more assurance, predeclare a small additional search before running it: several fixed seeds for MLP/RF; an L-BFGS MLP option; a modest LR C extension; and a limited comparison of F3/F4 versus F5 for deployment cost and stability. Keep every adaptive choice within participant-grouped inner validation. Mark same-cohort follow-ups as exploratory and rely on new participants for confirmation. Do not enlarge models merely to force a nonlinear winner.

For ESP32 deployment, LR is a reasonable candidate because ten linear class scores on twelve standardized inputs require few operations. The class-score stage contains 10×12 coefficients plus 10 intercepts, with preprocessing parameters as well. Actual end-to-end memory, latency and numeric parity still need measurement, particularly the two-second data window and feature calculations. This review did not implement or benchmark device inference.

## Wording suitable for the research report

“Across participant-held-out nested evaluation, logistic regression achieved the highest observed family-level accuracy (98.6%), followed closely by the small MLP (98.4%) and KNN, LDA and SVM (98.0%). The logistic-regression advantage over the MLP corresponded to one additional correct trial among 500 and does not establish reliable superiority. Adding gravity-direction descriptors substantially improved performance over flex-only representations. The complete inner model-selection procedure achieved 97.8% outer accuracy. A logistic-regression model using the F5 feature set was subsequently selected on all-participant development cross-validation and refitted for prospective testing.”

## Source files and methodological references

Dataset-specific findings are derived from the supplied files, chiefly `settings.json`, `accepted_trial_index.csv`, `calibration_references_used.csv`, the six `trial_features_*.csv` tables, `split_manifests/*.json`, `inner_cv_all_candidates.csv`, `outer_classifier_results.csv`, `outer_classifier_predictions.csv`, `outer_feature_set_classifier_results.csv`, `outer_feature_set_classifier_predictions.csv`, `outer_loso_results.csv`, `primary_outer_summary.csv`, `fit_warnings.csv`, `final_training_warnings.csv`, `final_all_participants_cv_best_params_per_feature_set.csv`, and `final_model_metadata.json`. All result paths refer to `smartglove_corrected_results/full_ed80fca0a01db881/` inside the uploaded results archive.

1. [scikit-learn 1.6.1 LogisticRegression](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html): regularization and multinomial classifier behavior.
2. [scikit-learn 1.6.1 MLPClassifier](https://scikit-learn.org/1.6/modules/generated/sklearn.neural_network.MLPClassifier.html): optimization settings and the small-dataset L-BFGS option.
3. [scikit-learn 1.6.1 nested versus non-nested cross-validation](https://scikit-learn.org/1.6/auto_examples/model_selection/plot_nested_cross_validation_iris.html): evaluation of selection procedures and optimistic selection scores.
4. [scikit-learn 1.6.1 cross-validation](https://scikit-learn.org/1.6/modules/cross_validation.html): grouped observations and participant-level separation.
5. [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html): model-specific importance and correlated predictors.

These references support general methodology. The precise explanations for model behavior are explicitly identified as interpretations of the supplied experiment, not universal guarantees.
