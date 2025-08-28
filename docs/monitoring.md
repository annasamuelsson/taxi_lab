# Monitoring-hook: vad & varför

En **monitoring-hook** är en liten kodbit som **automatiskt producerar övervakningssignal(er)** när din pipeline eller ditt API körs.
Syftet är att göra det enkelt att koppla på observability utan att bygga ett helt nytt system.

I den här demon finns två hooks:

1) **Träning → Evidently-rapport**  
   När `scripts/train.py` körs skapas en **Evidently Data Drift report** (HTML) där
   *referensdata* = träningsfeatures och *current* = testfeatures. Rapporten sparas i `artifacts/` och loggas som **MLflow artifact**.
   - Effekten: du får en snabb visuell kontroll av **drift i featuredistributioner** mellan tränings- och testset.

2) **Inferens-API → Prometheus-metrics**  
   `app/main.py` exponerar `/metrics` (Prometheus-format) samt mäter:
   - `predictions_total` (Counter): antal prediktioner
   - `prediction_latency_seconds` (Histogram): svarstid
   Dessa kan skrapas av **Prometheus** och visualiseras i **Grafana** (eller läsas via Azure Monitor/Managed Prometheus).

## Hur det används i praktiken
- **ML-teamet** tittar på Evidently-rapporten i MLflow efter träningsjobb för att se drift.
- **Ops/Platform-teamet** skrapar `/metrics` och larmar på t.ex. p95-latens eller avvikande trafik.

## Fortsättning
- Lägg till Evidently-rapporter mot *produktionsdata* (t.ex. batchade features/inference logs).
- Exportera prediktioner och labels till en "inference table" och bygg schemalagd driftanalys.
- Koppla `/metrics` till **Azure Monitor** eller **Managed Prometheus** i molnet.
