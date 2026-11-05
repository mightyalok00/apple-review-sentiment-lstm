.PHONY: help install test eval reproduce run workflow workflow-run docker-build docker-run health clean

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest
STREAMLIT ?= streamlit

help:
	@echo "Apple Review Intelligence & Sentiment RNN"
	@echo "Available commands:"
	@echo "  make install       Install all python requirements"
	@echo "  make test          Execute automated 18-test pytest suite with coverage"
	@echo "  make eval          Run baseline benchmarking, 5-fold CV & OOD stress-testing"
	@echo "  make reproduce     Execute 100% deterministic end-to-end reproduction runner"
	@echo "  make workflow      Run all 5 workflow pipeline stages"
	@echo "  make workflow-run  Run fast data partition, parquet sync, and test workflow"
	@echo "  make run           Launch Streamlit analytics dashboard locally"
	@echo "  make health        Run production container healthcheck"
	@echo "  make docker-build  Build production Docker image"
	@echo "  make docker-run    Run production Docker container"
	@echo "  make clean         Clean temporary python caches"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -p no:logfire tests/test_pipeline.py -v --cov=predict --cov=scraper --cov=workflow --cov=evaluate_model --cov-report=term

eval:
	$(PYTHON) evaluate_model.py

reproduce:
	$(PYTHON) reproduce.py

workflow:
	$(PYTHON) workflow.py --all

workflow-run:
	$(PYTHON) workflow.py --stages partition,parquet,test

run:
	$(STREAMLIT) run app.py

health:
	$(PYTHON) healthcheck.py

docker-build:
	docker build -t apple-sentiment:latest .

docker-run:
	docker run -d -p 8501:8501 --name apple_sentiment_app apple-sentiment:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage coverage.xml htmlcov .keras/matplotlib 2>/dev/null || true

