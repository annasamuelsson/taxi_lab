# Taxi Fare Service – Makefile
# Kör vanliga kommandon enklare

# 1. Installera beroenden
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

# 2. Köra tester
test:
	pytest -q

# 3. Träna modellen
train:
	python scripts/train.py --config configs/training.yaml

# 4. Starta API lokalt (Ctrl+C för att stoppa)
api:
	uvicorn app.main:app --reload --port 8080

# 5. Bygga och köra i Docker
docker-build:
	docker build -t taxi-fare-service -f docker/Dockerfile .

docker-run:
	docker run --rm -p 8080:8080 taxi-fare-service
