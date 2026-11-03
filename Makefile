.PHONY: install test eval-mock eval eval-bad baseline docker-eval

install:        ## install the package + test deps
	pip install -e ".[dev]"

test:           ## unit tests (no model needed)
	pytest -q

eval-mock:      ## fast smoke run with the deterministic mock provider
	python -m evalgate run --provider mock

eval:           ## real run against local Ollama (needs `ollama serve` + `ollama pull qwen2.5:1.5b`)
	python -m evalgate run --provider ollama

eval-bad:       ## demo: a worse prompt should FAIL the gate
	python -m evalgate run --provider ollama --prompt prompts/triage_v0_bad.txt

baseline:       ## record current Ollama metrics as the baseline (then commit baseline/)
	python -m evalgate run --provider ollama --update-baseline

docker-eval:    ## full pipeline in containers
	docker compose up --build --abort-on-container-exit --exit-code-from evalgate
