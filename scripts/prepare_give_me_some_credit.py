from pathlib import Path

import pandas as pd


RAW = Path("data/raw")
MAPPING = {
    "SeriousDlqin2yrs": "target",
    "RevolvingUtilizationOfUnsecuredLines": "TaxaDeUtilizacaoDeLinhasNaoGarantidas",
    "age": "Idade",
    "NumberOfTime30-59DaysPastDueNotWorse": "NumeroDeVezes30-59DiasAtrasoNaoPior",
    "DebtRatio": "TaxaDeEndividamento",
    "MonthlyIncome": "RendaMensal",
    "NumberOfOpenCreditLinesAndLoans": "NumeroDeLinhasDeCreditoEEmprestimosAbertos",
    "NumberOfTimes90DaysLate": "NumeroDeVezes90DiasAtraso",
    "NumberRealEstateLoansOrLines": "NumeroDeEmprestimosOuLinhasImobiliarias",
    "NumberOfTime60-89DaysPastDueNotWorse": "NumeroDeVezes60-89DiasAtrasoNaoPior",
    "NumberOfDependents": "NumeroDeDependentes",
}


def convert(source: str, target: str) -> None:
    df = pd.read_csv(RAW / source)
    df = df.rename(columns=MAPPING)
    df.to_csv(RAW / target, index=False)


if __name__ == "__main__":
    if (RAW / "cs-training.csv").exists():
        convert("cs-training.csv", "train.csv")
    if (RAW / "cs-test.csv").exists():
        convert("cs-test.csv", "test.csv")
