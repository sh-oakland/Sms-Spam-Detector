# Evaluating Traditional Machine Learning and LLM Strategies for SMS Spam Detection

**Oakland University | CSI-4130/5130 Artificial Intelligence Course Project**

> **Prerecorded project presentation:** **[ADD VIDEO URL HERE BEFORE NOVEMBER 30, 2026, 11:00 PM ET]**  
> The video should be viewable by anyone with the link (for example, an unlisted YouTube video or a publicly accessible Google Drive/OneDrive recording).  
> **Final report:** `report.pdf` — add the completed PDF to the repository root.  
> **Team member(s):** [ADD FULL NAME(S) AND OU EMAIL(S)]

## Project overview

Can a small instruction-following language model reliably identify unwanted SMS messages, and how does it compare with a conventional text classifier?

This project compares **five SMS classification methods** across two families: traditional machine learning (TF-IDF + Logistic Regression) and a pretrained large language model (**FLAN-T5-base**). The goal is not just to find a high-scoring model, but to understand **how each method works, why methods disagree, and how performance changes when text is deliberately obfuscated**.

Every method assigns one of two labels:

- **HAM:** a legitimate/non-spam message in the labeled research dataset.
- **SPAM:** a message labeled as spam in the dataset.

**This is a model-comparison and evaluation project. FLAN-T5 was not fine-tuned, and the classifiers are not intended to serve as a production security tool.**

## Research questions

1. How do traditional TF-IDF classifiers compare with zero-shot and few-shot FLAN-T5?
2. Does changing the prompt or the way FLAN-T5 chooses a label improve spam detection?
3. Are word-based and character-based features affected differently by misspelled or visually disguised spam terms?
4. Do the methods behave consistently on an independent test set and on modern, potentially ambiguous SMS examples?

## Data and evaluation design

- **Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), a historical, predominantly English-language labeled SMS dataset.
- **Cleaning:** remove exact duplicate messages (approximately **5,160 unique messages**).
- **Initial split:** stratified **75% training / 25% holdout** (approximately **3,870 / 1,290** messages).
- **Development evaluation:** **240 messages** (210 ham, 30 spam), used for E01–E03 method development.
- **Final evaluation:** **300 unseen messages** (263 ham, 37 spam), evaluated separately in E05.
- **Robustness testing:** compare original text with an altered version containing selected visual substitutions (for example, `free` → `fr33`). **48 development** and **38 final** messages were altered.
- **Metrics:** accuracy, spam-class precision, recall, F1-score, confusion matrices, and error analysis. The project emphasizes **spam F1** because the classes are imbalanced.

Training examples and few-shot prompt examples come from the training partition, not from the evaluation messages. The modern-message case study (E04) is illustrative and is **not** part of the labeled final-accuracy benchmark.

## The five methods

| Method | How it works | Why it is interesting |
|---|---|---|
| **Word TF-IDF + Logistic Regression** | Converts words and word pairs into numerical features and learns a supervised decision boundary. | Strong, fast, interpretable traditional baseline. |
| **Character TF-IDF + Logistic Regression** | Uses overlapping character sequences rather than complete words as features. | Can retain useful fragments when spelling changes. |
| **FLAN-T5 zero-shot** | Receives an instruction to classify an SMS, without labeled examples in the prompt. | Tests instruction following without task-specific training. |
| **FLAN-T5 few-shot** | Receives the instruction plus a small set of labeled demonstrations. | Tests whether in-context examples improve predictions. |
| **FLAN-T5 candidate-label scoring** | Scores predefined candidate outputs (`ham`, `spam`) instead of freely generating a label. | Tests the impact of the decision strategy using the same pretrained LLM. |

FLAN-T5 experiments use the open [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) checkpoint. The project changes prompts and inference strategies, **not its model weights**.

## Experiment progression: E01–E05

| Experiment | Question / change | Main lesson |
|---|---|---|
| **E01 — Initial baselines** | Compare word TF-IDF with initial FLAN-T5 zero-shot and few-shot prompts; add obfuscated inputs. | Word TF-IDF performed strongly; early FLAN-T5 prompts over-predicted spam. |
| **E02 — Prompt refinement** | Make instructions more explicit and revise few-shot examples. | Zero-shot spam F1 improved, but few-shot still produced excessive false alarms. |
| **E03 — Different methods** | Add candidate-label scoring and character TF-IDF. | How the LLM chooses a label matters; character features provided a strong additional baseline. |
| **E04 — Modern-message case study** | Explore 12 synthetic personal, political, phishing, and commercial SMS examples. | Model decisions can differ on contemporary or context-dependent wording. No verified labels, so **do not interpret as accuracy**. |
| **E05 — Independent final test** | Evaluate five final approaches on 300 unseen SMS messages; add paired bootstrap analysis and error examples. | Compare generalization, false positives/negatives, obfuscation behavior, and statistical uncertainty. |

### Selected development findings

On the original 240-message development sample, spam F1 increased from **0.227** to **0.482** for the initial versus improved FLAN-T5 zero-shot prompt. FLAN-T5 candidate-label scoring reached **0.731**, while character TF-IDF reached **0.947**. These numbers describe the **development sample**, not the final benchmark.

## Final results (E05)

**Independent final sample: 300 messages (263 ham, 37 spam).** Scores below are for the **spam class**, with higher F1 indicating a better balance of precision and recall.

| Classification method | Original spam F1 | Obfuscated spam F1 | False positives (original) | Missed spam (original) |
|---|---:|---:|---:|---:|
| Character TF-IDF + LR | **0.933** | 0.930 | 3 | 2 |
| Word TF-IDF + LR | 0.923 | **0.946** | 5 | 1 |
| FLAN-T5 candidate-label scoring | 0.781 | 0.700 | 2 | 12 |
| FLAN-T5 zero-shot (E02 prompt) | 0.624 | 0.556 | 27 | 8 |
| FLAN-T5 few-shot (E02 prompt) | 0.349 | 0.347 | 138 | 0 |

**Interpretation:**

- Both TF-IDF classifiers performed strongly on this historical labeled dataset; character TF-IDF had the highest **original-message** F1.
- The **character-versus-word F1 difference was small** (about 1.0 percentage point). The paired bootstrap 95% confidence interval was approximately **−3.1 to +5.0 percentage points**; it includes zero, so this experiment does **not** establish a reliable overall winner between those two classifiers.
- Few-shot FLAN-T5 caught all 37 labeled spam messages, but also incorrectly flagged **138 of 263 ham messages**. Strong recall alone is not enough for a useful filter.
- Candidate-label scoring improved on the tested zero-shot and few-shot FLAN-T5 approaches, but still missed more spam than the traditional baselines in the final evaluation.
- Obfuscation did not affect every method in the same way. The reported comparisons use paired original/altered messages; the changed-only results are available in the exported artifacts.

See `e05_final_holdout_metrics.csv`, `e05_confusion_matrices.png`, `e05_paired_model_comparisons.csv`, and `e05_error_examples.csv` in the experiment export for detailed evidence.

## Live demonstration

The Colab notebook includes a **batch SMS classification demo**. It runs the actual five classifiers on a small selection of editable messages and displays their predictions side by side.

The prepared examples cover:

1. An obfuscated prize offer.
2. A package-delivery payment request.
3. An unpaid-toll payment request.
4. A political event registration message.

The audience can compare where the methods agree and disagree. The messages can be edited before rerunning the cell; **predictions are calculated live, not hard-coded**. The demonstration also reports how many of the five classifiers predicted spam.

**Important:** These illustrative messages do not provide independently verified ground-truth labels. Agreement is **not** accuracy, and a count such as “5/5 spam votes” is **not** a calibrated confidence score. Sender authenticity and recipient consent cannot always be inferred from the SMS text alone.

## How to run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the final `SMS_Spam_Prompting_Project.ipynb` notebook from this repository (**File → Upload notebook**), or open it directly from GitHub.
3. Select **Runtime → Change runtime type → T4 GPU**, if available.
4. Choose **Runtime → Run all**. The notebook downloads the public UCI dataset and FLAN-T5 model weights and installs notebook-listed dependencies when needed. A first run requires an internet connection and may take longer because of the downloads and evaluation.
5. Review the E01–E05 evaluation tables, visualizations, statistical analysis, and error examples.
6. Find the **Live SMS Classification Demo** section. Modify the `DEMO_MESSAGES` dictionary to add or replace examples, then rerun that cell after the trained models and FLAN-T5 are loaded.
7. Use the final results-export cell to download the versioned `SMS_Spam_Project_Complete_Results_v###.zip` archive (actual version number depends on prior exports within the runtime).

**Runtime context:** The project was developed in Google Colab with a Tesla T4 GPU and Python 3.13. The code uses Python, PyTorch, Hugging Face Transformers, scikit-learn, Pandas, NumPy, Matplotlib, and SentencePiece. No paid inference API is required.

**Reproducibility:** Run cells in order in a fresh runtime, keep the notebook's fixed sampling/random-seed settings, and avoid interpreting isolated demo predictions as statistically representative. The exact package versions and model-cache state may influence runtime behavior.

## Experiment artifacts

The final notebook's export is intended to include the following files (subject to the version of the notebook committed):

| File | Contents |
|---|---|
| `development_all_methods.csv` | Development metrics across model variants |
| `development_all_predictions.csv` | Development message-level predictions |
| `development_changed_only.csv` | Results restricted to edited development messages |
| `e04_modern_sms_case_study.csv` | Qualitative modern SMS examples and model predictions |
| `e05_final_holdout_metrics.csv` | Final original/obfuscated evaluation metrics |
| `e05_final_holdout_predictions.csv` | Final message-level predictions |
| `e05_final_holdout_changed_only_metrics.csv` | Metrics for modified final messages |
| `e05_bootstrap_confidence_intervals.csv` | Confidence intervals for model F1 |
| `e05_paired_model_comparisons.csv` | Paired model F1 differences and uncertainty |
| `e05_error_examples.csv` | Representative classification errors |
| `development_vs_final_f1.png` | Development versus final performance |
| `e05_confusion_matrices.png` | False-positive / false-negative analysis |
| `e05_f1_confidence_intervals.png` | F1 uncertainty visualization |
| `RESULTS_README.txt` | Plain-language results summary |
| `manifest.json` | Exported file inventory |

The full experimental output archive does not need to be committed if the repository provides a runnable notebook and the required final report; include selected charts/tables as supporting materials if useful. Avoid committing redundant intermediate ZIPs.

## Limitations and future work

- **Historical data:** UCI SMS data does not necessarily reflect today's phishing and political-messaging patterns.
- **Small final sample:** Only 37 spam messages appear in the 300-message final set, so performance differences have uncertainty.
- **Limited robustness attacks:** A controlled substitution dictionary does not capture every way attackers disguise text.
- **Prompt sensitivity:** The FLAN-T5 results depend on the exact prompts, examples, candidate labels, and decoding/scoring procedures evaluated here.
- **Context matters:** A political reminder or commercial notification may be wanted or unwanted depending on consent and sender identity.
- **Potential benchmark contamination:** A publicly available dataset might overlap with some pretrained models' training data; this was not independently audited.

Future experiments could include **newer labeled SMS**, more diverse obfuscation attacks, additional models, supervised fine-tuning, calibration, and repeated runtime measurements on controlled hardware.

## Project submission and presentation

The course requires a **public GitHub repository** containing the finalized notebook/code, **report PDF**, and a README with the **prerecorded presentation link near the top**. Complete the repository and presentation sign-up by **Monday, November 30, 2026, 11:00 PM ET**.

The live presentations are scheduled for **December 1 and December 3, 2026**, via Zoom, with approximately **6 minutes presenting and 2 minutes for questions** per team. The prerecorded presentation is a required backup. Follow the instructor's separate instructions for the sign-up sheet and peer grading.

## References and acknowledgments

- Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. ACM DocEng. [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
- Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. [arXiv:2210.11416](https://arxiv.org/abs/2210.11416).
- Google. [FLAN-T5-base model card](https://huggingface.co/google/flan-t5-base) (Apache 2.0 license).
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. *JMLR, 12*, 2825–2830. [Paper](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html).
- [Hugging Face Transformers](https://github.com/huggingface/transformers), [PyTorch](https://pytorch.org/), and [Google Colab](https://colab.research.google.com/) documentation and software.

**AI-assistance disclosure:** OpenAI ChatGPT and Google Gemini were used as development/writing assistants where applicable. The submitted author(s) remain responsible for reviewing code, running experiments, interpreting outputs, and verifying all reported claims. Update this statement if necessary so it precisely reflects the assistance actually used.

**Team contributions:** [ADD CONTRIBUTIONS FOR EACH MEMBER, OR INDICATE THAT THIS WAS A SOLO PROJECT.]

---

### Final repository checklist

- [ ] Add full name(s) and Oakland University email(s).
- [ ] Replace the presentation-video placeholder and test the link in a private browser window.
- [ ] Commit the **final E01–E05 notebook** (not an early three-method draft).
- [ ] Add the completed report as `report.pdf`, or update its filename in this README.
- [ ] Verify all files and relative links in the public repository.
- [ ] Check that the notebook can be executed and the live demo runs as described.
- [ ] Complete the course presentation sign-up by November 30 at 11:00 PM ET.
