# Data

Source: Kaggle [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit).

Files expected in this folder:

- `train.csv`
- `test.csv`

The competition ships `cs-training.csv` and `cs-test.csv`; the preparation script renames them into the `train.csv` / `test.csv` contract used by the project.

## Download via the Kaggle API

```bash
mkdir -p data/raw
kaggle competitions download -c GiveMeSomeCredit -p data/raw
find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
```

Rename the files to match what the project expects:

```bash
python scripts/prepare_give_me_some_credit.py
```

Keep large files out of Git when possible and re-download them in Colab or the local environment as needed.
