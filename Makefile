.PHONY: help install dev test build deploy clean logs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Run in development mode
	python run.py

test: ## Run all tests in usual venv
	python -m pytest tests/ -v
	python test_api.py

dtest: ## Run all tests in docker venv
	docker-compose exec app pytest tests/ -v

build: ## Build Docker image
	docker-compose build

up: ## Start services
	docker-compose up -d

down: ## Stop services
	docker-compose down

logs: ## Show logs
	docker-compose logs -f app

clean: ## Clean up
	docker-compose down -v
	docker system prune -f
	rm -rf __pycache__ .pytest_cache vector_db/* local_storage/*

status: ## Check service status
	curl -s http://localhost:8000/api/v1/health | jq '.'