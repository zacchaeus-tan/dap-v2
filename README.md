# DAP: Emotion Classification on GoEmotions

Multi-label emotion classification of Reddit comments using the
[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
dataset (28 emotion labels). We started by evaluating a pretrained model as-is,
then moved to fine-tuning `roberta-base` and tuning its hyperparameters.

The notebooks are kept in the order we built them, so the repo also shows how
the approach evolved and what we learned at each step.

## Repository layout

```
data/
  raw/                 GoEmotions source files (goemotions_1-3.csv)
  processed/           goemotions_cleaned.csv (output of the EDA notebook)
notebooks/             Main pipeline, numbered in the order it was developed
experiments/           Individual team members' side experiments
  xinlei/
  zac/
results/
  roberta_v4/
    runs/              Per-epoch validation metrics for each training run
    plots/lr_sweep/    Micro/macro F1 for learning rates 2e-5, 3e-5, 4e-5
    plots/epoch_sweep/ F1 and validation loss for 3 vs 5 vs 7 epochs
```

## Learning progression

| # | Notebook | What changed | What we learned |
|---|----------|--------------|-----------------|
| 01 | `01_eda_goemotions.ipynb` | Merged the 3 raw files and removed duplicate text + subreddit rows (211,225 → 58,004 rows). Plotted the label distribution, text lengths, subreddit × emotion heatmap and labels per text | The labels are imbalanced and duplicated rows conflict, and `neutral` often co-occurs with other emotions, so it acts more like "mixed" than "no emotion" |
| 02 | `02_roberta_v1_pretrained_baseline.ipynb` | Ran the pretrained `pysentimiento/robertuito-emotion-analysis` model with no fine-tuning, scored with top-3 accuracy | A pretrained model works as a baseline, but this one is Spanish-focused and its labels don't match GoEmotions |
| 03 | `03_roberta_v2_zero_shot_with_splits.ipynb` | Same model, but evaluated on a proper 70/15/15 train/val/test split | Evaluate on held-out data. Next step: an English model, fine-tuned |
| 04 | `04_roberta_v3_finetune_roberta_base.ipynb` | Fine-tuned English `roberta-base` with a manual PyTorch training loop, sigmoid outputs for multi-label | How tokenization, multi-hot labels and the training loop fit together |
| 05 | `05_roberta_v4_trainer_hparam_tuning.ipynb` | Rewrote training with the Hugging Face `Trainer` (80/10/10 split), logged each run to CSV and swept learning rate and epochs | Validation loss bottoms out at epoch 2, but F1 keeps climbing until about epoch 6. LR 3e-5 to 4e-5 beat 2e-5. Tune only on the validation set and keep the test set for the final score |
| 06 | `06_bertopic_exploration.ipynb` | Learned BERTopic on 20 Newsgroups | Groundwork for topic modelling on our own data |

### Best result so far (RoBERTa V4)

`roberta-base`, LR 3e-5, batch size 16, 7 epochs, evaluated on the held-out test set:

| Metric | Score |
|--------|-------|
| F1 (micro) | 0.4125 |
| F1 (macro) | 0.3280 |

Validation F1 for every run is in `results/roberta_v4/runs/`, and the plots are in
`results/roberta_v4/plots/`.

## Experiments

- **xinlei/roberta_hf_datasets.ipynb**: fine-tunes RoBERTa using `datasets.Dataset` and `DataCollatorWithPadding`
- **xinlei/roberta_v3_variant.ipynb**: a variant of the V3 fine-tuning notebook
- **zac/bert_v1.ipynb → bert_v2.ipynb**: the same task with `bert-base-uncased`. v2 adds a train/val/test split and continues training from v1
- **zac/evaluation_metrics_wip.ipynb**: work in progress on a confusion matrix

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks from their own folder. Data paths are relative, for example
`../data/processed/goemotions_cleaned.csv`. Run `01_eda_goemotions.ipynb` first if
you need to regenerate the cleaned dataset.

Fine-tuning works best on a GPU, such as Colab. On CPU it is very slow.
