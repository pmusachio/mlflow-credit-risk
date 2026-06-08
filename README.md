# MLflow Credit Risk

Projeto de ciencia de dados baseado no desafio/dataset Kaggle [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit).

## 1. Business Problem

Uma area de credito precisa classificar risco de inadimplencia com rastreabilidade de experimentos.

**Objetivo:** Classificar risco de inadimplencia com pipeline reprodutivel, metricas versionadas e opcao de tracking com MLflow.

**Metrica principal:** ROC AUC, average precision, F1 e registro opcional no MLflow

## 2. Business Assumptions

- A decisao de credito exige metricas reproduziveis e historico de modelos.
- Falsos negativos podem aumentar inadimplencia; falsos positivos podem reduzir aprovacao boa.
- MLflow ajuda a comparar versoes de modelos e parametros.

## 3. Solution Strategy

1. **Step 01. Data Description:** Validar schema, dimensoes, nulos, tipos e granularidade.
2. **Step 02. Feature Engineering:** Criar variaveis orientadas ao dominio do desafio.
3. **Step 03. Data Filtering:** Remover registros sem utilidade analitica ou com risco de vazamento.
4. **Step 04. Exploratory Data Analysis:** Validar hipoteses e separar sinais relevantes de ruido.
5. **Step 05. Data Preparation:** Imputar, escalar e codificar variaveis para modelagem.
6. **Step 06. Feature Selection:** Separar IDs, alvo, variaveis de entrada e colunas descartadas.
7. **Step 07. Machine Learning Modelling:** Treinar baseline reprodutivel e avaliar metricas tecnicas.
8. **Step 08. Hyperparameter/Fit Strategy:** Reservar espaco para tuning, threshold e comparacao de modelos.
9. **Step 09. Business Translation:** Converter metricas em decisao, priorizacao, risco, receita ou operacao.
10. **Step 10. Delivery:** Versionar experimentos no MLflow, promover modelo para Production e documentar API/SageMaker.

## 4. Data Source

Fonte Kaggle: [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit).

Arquivos esperados:

- `data/raw/train.csv`
- `data/raw/test.csv`

- A competicao disponibiliza `cs-training.csv` e `cs-test.csv`; o script de preparo gera `train.csv` e `test.csv` no contrato do projeto.

## 5. Development Journey

Os notebooks foram organizados para mostrar a evolucao da analise, desde o entendimento do problema ate a traducao do resultado para negocio. Eles sao os melhores pontos para tirar screenshots do portfolio:

- `notebooks/00_business_understanding.ipynb`
- `notebooks/01_data_understanding.ipynb`
- `notebooks/02_exploratory_analysis.ipynb`
- `notebooks/03_feature_engineering.ipynb`
- `notebooks/04_modeling_and_business_results.ipynb`
- `notebooks/05_deployment_and_consumption.ipynb`

## 6. Top Data Insights and Hypotheses

- Atrasos anteriores sao sinais fortes de risco futuro.
- Taxa de utilizacao de credito e endividamento ajudam a separar perfis.
- Renda mensal ausente precisa ser tratada sem descartar clientes.

## 7. Model or Analysis Applied

Classificacao binaria com pipeline sklearn e registro opcional no MLflow.

## 8. Performance and Business Results

Perfil de dados reproduzido em `reports/data_profile.json`: 150000 linhas e 12 colunas analisadas.

Principais saidas do pipeline:

- `reports/metrics.json`
- `models/model.joblib`
- `mlruns/`

## 9. Business Translation

O score apoia politica de credito com cortes por risco e trilha auditavel de experimentos.

## 10. Repository Structure

- `configs/project.toml`: contrato do projeto, dados, alvo, metricas e parametros.
- `src/mlflow_credit_risk/`: codigo Python modular para dados, features, modelagem, analise e consumo.
- `notebooks/`: jornada analitica em notebooks.
- `data/raw/`: arquivos baixados do Kaggle ou base analitica preparada.
- `reports/`: metricas, perfis e resultados gerados pela execucao.
- `docs/deployment.md`: notas objetivas de entrega e consumo.
- `scripts/sagemaker_deployment_template.py`: template para deploy MLflow em SageMaker.
- `models/`: modelo treinado quando aplicavel.

## 11. Como executar no Google Colab

1. Abra um notebook novo no Google Colab.
2. Gere seu token em Kaggle > Account > API > Create New Token.
3. Rode as celulas abaixo.

Clone o repositorio e instale as dependencias:

```python
REPO_URL = "https://github.com/<seu-usuario>/<nome-do-repositorio>.git"
!git clone {REPO_URL} project
%cd project
!python -m pip install -q -r requirements.txt
```

Baixe ou prepare os dados:

```python
from google.colab import files
files.upload()  # envie o arquivo kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!mkdir -p data/raw
!python -m pip install -q kaggle
!kaggle competitions download -c GiveMeSomeCredit -p data/raw
!find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
!python scripts/prepare_give_me_some_credit.py
```

Execute o fluxo principal:

```python
!PYTHONPATH=src python -m mlflow_credit_risk.cli validate-config
!PYTHONPATH=src python -m mlflow_credit_risk.cli profile
!PYTHONPATH=src python -m mlflow_credit_risk.cli train
```

## 12. Execucao local

Para rodar localmente em modo batch:

```bash
git clone <URL_DO_REPOSITORIO> project
cd project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m mlflow_credit_risk.cli profile
PYTHONPATH=src python -m mlflow_credit_risk.cli train
```

### Execucao local para MLflow e API

O treino basico roda no Colab. Use o ambiente local quando quiser acompanhar experimentos no MLflow UI ou manter a API online.

```bash
git clone <URL_DO_REPOSITORIO> project
cd project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-mlflow.txt -r requirements-api.txt
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Em outro terminal:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m mlflow_credit_risk.cli train
PYTHONPATH=src uvicorn mlflow_credit_risk.api:app --reload
PYTHONPATH=src python scripts/sample_api_request.py
```

Abra a interface em `http://127.0.0.1:5000`. Dentro do MLflow, selecione o experimento `08_mlflow_credit_risk`, abra a run mais recente e confira parametros, metricas e artefatos do modelo.

### SageMaker

Depois de registrar o melhor modelo no MLflow e promover para `Production`, gere a configuracao de deploy:

```bash
python scripts/sagemaker_deployment_template.py --model-uri models:/mlflow-credit-risk/Production
```

O script fica em modo dry run por padrao para evitar criacao acidental de infraestrutura. Use `--execute` apenas com credenciais AWS, role SageMaker e tracking URI configurados.

### Alternativas mais baratas que AWS

Para portfolio, a opcao mais economica e manter o MLflow local ou em DagsHub e publicar apenas a API FastAPI:

- **DagsHub:** tracking remoto para runs do MLflow sem manter servidor proprio.
- **Render, Railway ou Fly.io:** deploy simples da API com custo baixo para demo.
- **Google Cloud Run:** API com scale-to-zero e cobranca por uso.
- **Hugging Face Spaces:** demo publica usando Docker ou app visual.

Minha recomendacao para este projeto: MLflow local/DagsHub para evidenciar experimentos e Render ou Cloud Run para expor `/predict`. SageMaker fica como alternativa enterprise, nao como caminho padrao de menor custo.

Veja os detalhes em `docs/deployment.md`.

## 13. Next Steps to Improve

- Adicionar matriz de decisao por faixas de score.
- Comparar modelos com tracking no MLflow UI.
- Criar validacao temporal para simular safras de credito.

## 14. Tests

```bash
python -m pytest
```
