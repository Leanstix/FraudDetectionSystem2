from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.pipeline.orchestrator import FraudPipeline


def infer_dataset_name(reference_path: str, target_path: str) -> str:
    joined = f"{Path(reference_path).stem}_{Path(target_path).stem}".lower().replace(" ", "_").replace("+", "_")
    tokens = [token for token in joined.replace("-", "_").split("_") if token]
    if "truman" in joined:
        return "the_truman_show"
    return "_".join(tokens)


def run_inspect_pair(reference_path: str, input_path: str, config_path: str | None = None) -> dict:
    settings = Settings.from_env_and_file(config_path)
    dataset_name = infer_dataset_name(reference_path, input_path)
    pipeline = FraudPipeline(
        settings=settings,
        reference_path=reference_path,
        target_path=input_path,
        output_path="outputs/inspect_only.txt",
        dataset_name=dataset_name,
        no_llm=True,
        verbose=False,
    )
    report = pipeline.inspect()

    ref = report["reference"]
    tgt = report["target"]
    print(f"dataset name: {dataset_name}")
    print(f"reference path: {reference_path}")
    print(f"target path: {input_path}")
    print(f"reference transactions: {ref['transactions']}")
    print(f"target transactions: {tgt['transactions']}")
    print(f"reference users: {ref['users']}")
    print(f"target users: {tgt['users']}")
    return report


def run_predict_pair(
    reference_path: str,
    input_path: str,
    output_path: str,
    no_llm: bool = False,
    verbose: bool = False,
    config_path: str | None = None,
):
    settings = Settings.from_env_and_file(config_path)
    dataset_name = infer_dataset_name(reference_path, input_path)
    pipeline = FraudPipeline(
        settings=settings,
        reference_path=reference_path,
        target_path=input_path,
        output_path=output_path,
        dataset_name=dataset_name,
        no_llm=no_llm,
        verbose=verbose,
    )
    artifacts = pipeline.run()
    pipeline.print_summary(artifacts)
    return artifacts
