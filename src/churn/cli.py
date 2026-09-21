"""Единая точка входа CLI.
Примеры:
    python -m churn.cli generate-data
    python -m churn.cli eda
    python -m churn.cli train
    python -m churn.cli evaluate
    python -m churn.cli ab-test --control-engaged 420 --control-total 5000 \
                                --treat-engaged 520 --treat-total 5000
    python -m churn.cli power --baseline-rate 0.08 --mde 0.25
    python -m churn.cli pipeline
"""
from __future__ import annotations
import argparse
import json
from churn.config import load_config
from churn.utils.logger import get_logger
log = get_logger("churn.cli")
def _cmd_generate_data(args) -> None:
    from churn.data.make_dataset import build_raw_dataset
    cfg = load_config(args.config)
    build_raw_dataset(cfg, force_synthetic=not args.from_sources)
def _cmd_eda(args) -> None:
    from churn.pipelines.training_pipeline import run_eda_step
    cfg = load_config(args.config)
    print(json.dumps(run_eda_step(cfg), ensure_ascii=False, indent=2, default=str))
def _cmd_train(args) -> None:
    from churn.pipelines.training_pipeline import run_train_step
    cfg = load_config(args.config)
    result = run_train_step(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
def _cmd_evaluate(args) -> None:
    from churn.models.registry import load_meta
    cfg = load_config(args.config)
    print(json.dumps(load_meta(cfg), ensure_ascii=False, indent=2, default=str))
def _cmd_predict(args) -> None:
    from churn.models.predict import score_file
    cfg = load_config(args.config)
    result = score_file(args.input, args.output, cfg)
    print(result.head(20).to_string(index=False))
def _cmd_ab_test(args) -> None:
    from churn.ab_testing.ab_test import proportion_test, summary_table
    cfg = load_config(args.config)
    result = proportion_test(
        control_engaged=args.control_engaged,
        control_total=args.control_total,
        treat_engaged=args.treat_engaged,
        treat_total=args.treat_total,
        alpha=cfg.ab_testing.alpha,
    )
    print(summary_table(result))
def _cmd_power(args) -> None:
    from churn.ab_testing.power_analysis import sample_size_proportions
    cfg = load_config(args.config)
    result = sample_size_proportions(
        baseline_rate=args.baseline_rate,
        mde_relative=args.mde,
        alpha=cfg.ab_testing.alpha,
        power=cfg.ab_testing.power,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
def _cmd_pipeline(args) -> None:
    from churn.pipelines.training_pipeline import run_full_pipeline
    cfg = load_config(args.config)
    print(json.dumps(run_full_pipeline(cfg), ensure_ascii=False, indent=2, default=str))
def _cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churn", description="Customer Churn Prediction System (Classical ML & A/B)"
    )
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate-data", help="собрать/сгенерировать сырой датасет")
    p.add_argument("--from-sources", action="store_true", help="использовать API/БД вместо синтетики")
    p.set_defaults(func=_cmd_generate_data)
    sub.add_parser("eda", help="построить отчёт EDA").set_defaults(func=_cmd_eda)
    sub.add_parser("train", help="обучить модели").set_defaults(func=_cmd_train)
    sub.add_parser("evaluate", help="показать метрики лучшей модели").set_defaults(func=_cmd_evaluate)
    sub.add_parser("pipeline", help="полный сквозной прогон").set_defaults(func=_cmd_pipeline)
    p = sub.add_parser("predict", help="батч-скоринг файла")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_cmd_predict)
    p = sub.add_parser("ab-test", help="оценить A/B-тест удержания")
    p.add_argument("--control-engaged", type=int, required=True)
    p.add_argument("--control-total", type=int, required=True)
    p.add_argument("--treat-engaged", type=int, required=True)
    p.add_argument("--treat-total", type=int, required=True)
    p.set_defaults(func=_cmd_ab_test)
    p = sub.add_parser("power", help="расчёт MDE / размера выборки")
    p.add_argument("--baseline-rate", type=float, default=0.08)
    p.add_argument("--mde", type=float, default=0.25)
    p.set_defaults(func=_cmd_power)
    p = sub.add_parser("serve", help="запустить FastAPI-сервис")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=_cmd_serve)
    return parser
def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
if __name__ == "__main__":  # pragma: no cover
    main()
