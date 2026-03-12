# mAceReason-Math: A Dataset of High-quality Multilingual Math Problems Ready for RLVR

## Dataset Description

This dataset contains translations of mathematical reasoning problems from [AceReason-Math](https://huggingface.co/datasets/nvidia/AceReason-Math), a mathematical reasoning dataset curated for RLVR training.
The dataset covers 14 languages and contains a total of ~140k translated reasoning problems and answers.
We release this dataset to facilitate multilingual RLVR research in the research community.

### Key Features

- **Crosslingually Parallel**: the `train` and `test` splits contain samples available across all languages
- **Data Cleaning**: English source samples processed to remove artifacts before translation
- **Translation Method**: LLM-based translation (Claude Sonnet 4) with human validation of test set in 11 languages
- **Languages**:
  - _with human review_: Chinese, Spanish, German, French, Russian, Brazilian Portuguese, Italian, Japanese, Korean, Thai + English (original)
  - _additionally provided without human review_: Swahili, Telugu, Bengali

## Usage

**Requirements:**

```bash
pip install datasets bsdiff4
```

**Loading the Dataset:**

We provide the snippet below to use our data with Huggingface `datasets`.
Load any released language directly with `load_macereason_math(lang)`. For most languages, this reads the JSONL files from GitHub. For `lang="en"`, it reconstructs the cleaned English version from the released modifications and the original [AceReason-Math](https://huggingface.co/datasets/nvidia/AceReason-Math) dataset.

```python
from datasets import load_dataset
from typing import Literal
import base64, bsdiff4

def load_macereason_math(lang: Literal["bn", "de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "sw", "te", "th", "zh"]):
    """Load mAceReason-Math; lang='en' reconstructs cleaned English."""
    base_url = "https://raw.githubusercontent.com/apple/ml-macereason-math/main"
    splits = ("train", "test", "train_all", "asy")
    if lang == "en":
        mods = load_dataset("json", data_files={s: f"{base_url}/en_modifications/{s}.jsonl" for s in splits})
        original = load_dataset("nvidia/AceReason-Math", split="train", revision="a5cc41c5ecfc1d4a6571f98ade92d7fec100b2a8")

        def reconstruct(mod):
            orig = original[mod["original_idx"]]
            problem = orig["problem"] if not mod["english_problem_modification"] else bsdiff4.patch(orig["problem"].encode(), base64.b64decode(mod["english_problem_modification"])).decode()
            solution = orig["answer"] if not mod["english_solution_modification"] else bsdiff4.patch(orig["answer"].encode(), base64.b64decode(mod["english_solution_modification"])).decode()
            return {
                "original_idx": mod["original_idx"],
                "problem": problem,
                "solution": solution,
                "english_has_been_cleaned": mod["english_has_been_cleaned"],
            }

        return mods.map(reconstruct, remove_columns=["english_problem_modification", "english_solution_modification"])
    return load_dataset("json", data_files={s: f"{base_url}/{lang}/{s}.jsonl" for s in splits})
```

Example:

```python
macereason_de = load_macereason_math("de")
macereason_en = load_macereason_math("en")

print(macereason_de["train"][0], macereason_de["train_all"], macereason_de["test"])
print(macereason_en["train"][0], macereason_en["train_all"], macereason_en["test"])
```

## Dataset Structure

### Data Instances

**Translation Configurations (e.g., German):**

```json
{
  "original_idx": 21193,
  "problem": "Ein Parallelogramm hat 3 seiner Eckpunkte bei (1,2), (3,8) und (4,1). Berechne die Summe der möglichen x-Koordinaten für den 4. Eckpunkt.",
  "solution": "8",
  "english_has_been_cleaned": false
}
```

**English Modifications Configuration (`en_modifications`):**

```json
{
  "original_idx": 193,
  "english_problem_modification": "QlNESUZGNDA...",
  "english_solution_modification": null,
  "english_has_been_cleaned": true
}
```

The English modifications contain base64-encoded binary patches (bsdiff4 format) that can be applied to the original [AceReason-Math](https://huggingface.co/datasets/nvidia/AceReason-Math) dataset to reconstruct the cleaned English version. See [Usage](#usage) for details.

### Data Fields

**Translation Configurations:**
- **original_idx**: Unique identifier from the original dataset
- **problem / solution**: The mathematical problem statement and solution (in target language)
- **english_has_been_cleaned**: Boolean indicating if the English source was cleaned of artifacts before translation

**English Modifications Configuration (`en_modifications`):**
- **original_idx**: Unique identifier from the original dataset
- **english_problem_modification**: Base64-encoded bsdiff4 patch for the problem (null if unchanged)
- **english_solution_modification**: Base64-encoded bsdiff4 patch for the solution (null if unchanged)
- **english_has_been_cleaned**: Boolean indicating if the sample was cleaned

### Data Splits

- **`train`**: 7,620 samples per language — parallel across all languages
- **`test`**: 190 samples per language — parallel
- **`train_all`**: All available samples per language (varies by language from 10,270 to 12,245)
- **`asy`**: 96 samples per language — separate split with specific problem sets containing `[asy]` for diagrams. This split is provided separately, as this tests a very specific model skill.

The `test` split is randomly sampled and consistent across all languages. The `train` split is parallel across all languages, while the `train_all` split contains all available data per language (except for the `test` and `asy` splits).

## Language Statistics

| Language   | Code | `train` (parallel) | `test` (parallel) | `train_all` |
| ---------- | ---- | ------------------ | ----------------- | ----------- |
| English    | en   | 7,620              | 190               | 12,245      |
| German     | de   | 7,620              | 190               | 11,151      |
| French     | fr   | 7,620              | 190               | 11,007      |
| Spanish    | es   | 7,620              | 190               | 11,346      |
| Chinese    | zh   | 7,620              | 190               | 10,470      |
| Russian    | ru   | 7,620              | 190               | 11,237      |
| Japanese   | ja   | 7,620              | 190               | 10,376      |
| Thai       | th   | 7,620              | 190               | 11,104      |
| Portuguese | pt   | 7,620              | 190               | 10,632      |
| Italian    | it   | 7,620              | 190               | 10,646      |
| Korean     | ko   | 7,620              | 190               | 10,270      |
| Swahili    | sw   | 7,620              | 190               | 11,124      |
| Telugu     | te   | 7,620              | 190               | 10,964      |
| Bengali    | bn   | 7,620              | 190               | 11,082      |

English data can be reconstructed from the original [AceReason-Math](https://huggingface.co/datasets/nvidia/AceReason-Math) dataset using the `en_modifications` config. See [Usage](#usage).

## Dataset Creation

### Source Data

The source data comes from [AceReason-Math](https://huggingface.co/datasets/nvidia/AceReason-Math), a mathematical reasoning dataset curated for RLVR training. We first filter the original English data by removing problematic samples, such as problems which already reveal the solution in the problem statement or which contain critical references to diagrams or figures, which are not provided. This affects roughly 4% of the original data. We also clean samples with minor issues such as task number annotations ("Problem 4.1: [...]"), in ~11% of samples. This filtering and cleaning process is conducted using Claude Sonnet 4.

### Translation Process

The translations are also conducted using Claude Sonnet 4. We initially translate 100 random samples and solicit feedback from our native speaker annotators. We then use this feedback to improve our prompts and run the translation pipeline for the entire dataset. We use an iterative approach where the translations are graded against predefined rubrics in a LLM-as-a-Judge rating round. If any issues are detected, we retranslate the sample with additional feedback. This process is repeated up to 5 times.

**Note:** In the translations, we localize number formats (e.g. US: `1,000,000.0` vs. German: `1.000.000,0`) in the problems and solutions. In simple cases, most symbolic verifiers (such as [`huggingface/math-verify`](https://github.com/huggingface/Math-Verify)) can handle this but might fail for more complex cases, which only support the US format. In those cases, you may want to use the English solution.

For more details, please refer to our accompanying [paper](https://arxiv.org/todo).

## License/Terms of Use

This dataset is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC-by-NC-ND 4.0) available at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.

## Intended Use

The mAceReason-Math Dataset is intended to be used by the community for multilingual reinforcement learning with LLMs. The data may be used to train and evaluate.

## Release Date

March 2026

## Correspondence to

Konstantin Dobler (konstantin.dobler@hpi.de) and Simon Lehnerer (simon.lehnerer@apple.com).

## Citation

```bibtex
@dataset{mAceReasonMath,
  title={mAceReason-Math: A Dataset of High-quality Multilingual Math Problems Ready for RLVR},
  author={Konstantin Dobler and Simon Lehnerer and Federico Scozzafava and Jonathan Janke and Mohamed Ali},
  year={2026},
}
```

## Acknowledgments

We thank the AceReason-Math authors for compiling the original dataset on which we base our translations. We also thank our team of professional translators and native speaker validators who ensured high-quality localizations in their respective languages.
