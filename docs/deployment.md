# Entrega e consumo

O projeto foi estruturado para versionar experimentos, promover um modelo para Production e expor predicoes por API. SageMaker fica como opcao corporativa, mas nao e a opcao mais economica para portfolio.

## Canais

- **MLflow Tracking:** comparacao de runs, parametros e metricas.
- **MLflow Model Registry:** promocao do melhor modelo para `Production`.
- **FastAPI:** endpoint `/predict` para consumo local ou cloud.
- **SageMaker:** `scripts/sagemaker_deployment_template.py` com configuracao base para deploy do modelo registrado.
- **Alternativas de baixo custo:** DagsHub para tracking remoto, Render/Railway/Fly.io/Cloud Run/Hugging Face Spaces para publicar a API ou uma demo.

## Como ver a interface do MLflow

Use dois terminais a partir da raiz do projeto.

Terminal 1: iniciar a interface.

```bash
python -m pip install -r requirements.txt -r requirements-mlflow.txt
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Abra no navegador:

```text
http://127.0.0.1:5000
```

Terminal 2: gerar uma run rastreada.

```bash
PYTHONPATH=src python -m mlflow_credit_risk.cli train
```

Na interface, selecione o experimento `08_mlflow_credit_risk`, abra a run mais recente e confira:

- **Parameters:** configuracoes do projeto, alvo e familia do problema.
- **Metrics:** ROC AUC, average precision, F1 e demais metricas registradas.
- **Artifacts:** arquivo `models/model.joblib` e artefatos do modelo sklearn.

Se a run nao aparecer, confirme que o comando de treino foi executado na raiz do repositorio e que a UI esta apontando para `./mlruns`.

## API local

```bash
python -m pip install -r requirements.txt -r requirements-api.txt
PYTHONPATH=src uvicorn mlflow_credit_risk.api:app --reload
```

## SageMaker

```bash
python scripts/sagemaker_deployment_template.py --model-uri models:/mlflow-credit-risk/Production
```

O script roda em modo dry run por padrao. Use `--execute` somente depois de configurar credenciais AWS, role SageMaker e tracking URI.

## Alternativas mais baratas que AWS

Para portfolio, a recomendacao pratica e separar tracking e inferencia:

- **Tracking local:** custo zero; rode `mlflow ui` localmente e mostre screenshots no README.
- **DagsHub + MLflow:** bom para tracking remoto sem manter servidor proprio. Configure `MLFLOW_TRACKING_URI`, usuario e token do DagsHub, depois rode o mesmo comando de treino.
- **Render:** simples para publicar a API FastAPI com plano gratuito/baixo custo, bom para demos que podem dormir quando ficam inativas.
- **Railway:** bom para deploy rapido de API com variaveis de ambiente e logs simples.
- **Fly.io:** bom quando voce quer Docker e maquina pequena com custo previsivel.
- **Google Cloud Run:** bom para API com scale-to-zero e cobranca por uso.
- **Hugging Face Spaces:** bom para demo publica, especialmente se transformar a API em app/demo Docker ou Streamlit.

Recomendacao de custo para este projeto: mantenha a UI do MLflow local ou no DagsHub e publique apenas a API em Render, Cloud Run ou Fly.io. Use SageMaker somente se o objetivo for demonstrar deploy gerenciado em stack AWS.

Exemplo de tracking remoto com DagsHub:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<usuario>/<repositorio>.mlflow"
export MLFLOW_TRACKING_USERNAME="<usuario>"
export MLFLOW_TRACKING_PASSWORD="<token>"
PYTHONPATH=src python -m mlflow_credit_risk.cli train
```

Links uteis para checar planos atuais:

- [MLflow Tracking UI](https://mlflow.org/docs/latest/ml/tracking/)
- [DagsHub MLflow](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
- [Render Pricing](https://render.com/pricing)
- [Railway Pricing](https://railway.com/pricing)
- [Fly.io Pricing](https://fly.io/docs/about/pricing/)
- [Google Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Hugging Face Spaces Pricing](https://huggingface.co/pricing)
