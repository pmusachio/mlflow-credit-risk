# Dados

Fonte Kaggle: [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit).

Arquivos esperados nesta pasta:

- `train.csv`
- `test.csv`

- A competicao disponibiliza `cs-training.csv` e `cs-test.csv`; o script de preparo gera `train.csv` e `test.csv` no contrato do projeto.

## Download via Kaggle API

```bash
mkdir -p data/raw
kaggle competitions download -c GiveMeSomeCredit -p data/raw
find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
```

Ajuste de nomes esperado pelo projeto:

```bash
python scripts/prepare_give_me_some_credit.py
```

Mantenha arquivos grandes fora do Git quando necessario e baixe-os novamente no Colab ou no ambiente local.
