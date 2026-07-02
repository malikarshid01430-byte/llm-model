install:
	python -m pip install -r requirements.txt

train:
	python train.py

evaluate:
	python -m pytest -q tests

generate:
	python generate.py

run-api:
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
