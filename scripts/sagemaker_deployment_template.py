"""Template for deploying the MLflow production model to SageMaker.

The default mode prints the deployment configuration. Use --execute only
after AWS credentials, IAM role and MLflow tracking are configured.
"""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an MLflow SageMaker deployment.")
    parser.add_argument("--app-name", default="mlflow-credit-risk")
    parser.add_argument("--model-uri", default="models:/mlflow-credit-risk/Production")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "app_name": args.app_name,
        "model_uri": args.model_uri,
        "region_name": args.region,
        "mode": "create",
        "execution_role_arn": "<AWS_SAGEMAKER_EXECUTION_ROLE_ARN>",
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
    }

    print(json.dumps(config, indent=2))
    if not args.execute:
        print("Dry run only. Re-run with --execute after replacing the execution_role_arn value.")
        return

    import mlflow.deployments

    if config["execution_role_arn"].startswith("<"):
        raise ValueError("Set execution_role_arn before executing the deployment.")

    client = mlflow.deployments.get_deploy_client("sagemaker")
    client.create_deployment(
        name=args.app_name,
        model_uri=args.model_uri,
        config=config,
    )


if __name__ == "__main__":
    main()
