.PHONY: help install data eda train evaluate ab-test power demo serve pipeline test lint clean
PYTHON ?= python
export PYTHONPATH := src
help:
	@echo "Customer Churn Prediction System — make targets"
	@echo "  install    установить зависимости"
	@echo "  data       сгенерировать демо-датасет"
	@echo "  eda        отчёт EDA"
	@echo "  train      обучение + подбор гиперпараметров (MLflow)"
	@echo "  evaluate   оценка лучшей модели"
	@echo "  ab-test    пример A/B-теста"
	@echo "  power      расчёт MDE / размера выборки"
	@echo "  demo       быстрый сквозной демо-прогон"
	@echo "  serve      FastAPI инференс"
	@echo "  pipeline   полный сквозной прогон"
	@echo "  test       pytest"
	@echo "  clean      очистка артефактов"
install:
	$(PYTHON) -m pip install -r requirements.txt
data:
	$(PYTHON) -m churn.cli generate-data
eda:
	$(PYTHON) -m churn.cli eda
train:
	$(PYTHON) -m churn.cli train
evaluate:
	$(PYTHON) -m churn.cli evaluate
ab-test:
	$(PYTHON) -m churn.cli ab-test \
		--control-engaged 420 --control-total 5000 \
		--treat-engaged 520 --treat-total 5000
power:
	$(PYTHON) -m churn.cli power --baseline-rate 0.08 --mde 0.25
demo:
	$(PYTHON) scripts/run_demo.py
serve:
	$(PYTHON) -m uvicorn api.app:app --reload --port 8000
pipeline:
	$(PYTHON) -m churn.cli pipeline
test:
	$(PYTHON) -m pytest -q
clean:
	rm -rf artifacts mlruns reports .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
