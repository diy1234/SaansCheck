# SaansCheck

**Delhi-NCR Air Quality Intelligence and Personalized Risk Advisor**  
MCA Minor Project · Semester 3

SaansCheck is a Streamlit dashboard for exploring historical air-quality data, comparing Delhi-NCR locations, viewing model forecasts and evaluation results, and inspecting separately sourced recent observations.

## Dashboard

- Historical Intelligence: overview findings, historical trends, location rankings, pollutant and seasonal analytics, a daily AQI calendar, health context, and model insights.
- Recent Air Quality: the fresh-session landing page. It requests optional browser geolocation through the native permission prompt, then uses WAQI's server-side geolocalized feed only when station identity and source coordinates validate and the station is within 25 km. Coordinates are used for that uncached lookup and are cleared from the component result afterward. If permission or matching fails, users can select a city and WAQI station manually; observations retain source timestamps, retrieval time, explicit scale, and unavailable/stale states. The primary panel attempts to show only a fresh, directly reported CPCB National AQI matched to the exact selected station; it stays unavailable unless the official OGD response schema and overall-AQI field are explicitly verified and configured. WAQI remains a separate US-scale alternative. The source comparison links to AQI.in without retrieving or representing AQI.in values. The Forecast page shows observations separately and leaves predictions unavailable until authorized station history and a compatible, independently validated model are available.
- AQI health guidance: for fresh, valid WAQI observations only, a compact panel shows categories and general precautions from the provider's documented US EPA 2016 scale: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, and 301+ Hazardous. Guidance is informational, not medical advice; it is not shown as current when the AQI or freshness validation is missing.
- Health alerts: a transient five-second cloud overlay and a dismissible side card for fresh WAQI observations at AQI 101 or above on the provider's US EPA 2016 scale (101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous). Stale readings and model predictions do not trigger health alerts.
- Forecasts: saved Linear Regression and Random Forest models for Day+1, Day+2, and Day+3. The forecast page identifies its historical origin and must not be interpreted as a current forecast.
- Appearance: Dark is the default; the sidebar Appearance control switches between Dark and Light themes. Selection and dashboard filters persist across Streamlit reruns in the session.
- Intro: a local four-second CSS/HTML welcome animation runs once per fresh Streamlit session. It is not an MP4. A separate compact sidebar emblem animation also runs once per session. Both respect reduced-motion preferences.

Historical dataset AQI, recent WAQI observations, and model predictions remain distinct. Historical AQI is dataset-defined; its calculation method has not been verified as CPCB AQI, and historical pollutant units are not verified. WAQI documents its reported AQI on the US EPA 2016 scale, which is not the Indian CPCB/NAQI scale. The historical dataset covers Delhi, Faridabad, Ghaziabad, Gurgaon, and Noida through 2023-12-31.

## Technology

- Python 3.10 (the project environment is Python 3.10.11)
- Streamlit 1.65.0 in the inspected environment
- pandas, NumPy, scikit-learn, joblib, Matplotlib, and Seaborn
- unittest and Streamlit `AppTest`

Dependency versions are not pinned in `requirements.txt`.

## Architecture

```text
app.py                         Streamlit entry point, navigation, and page rendering
src/data_loader.py             Cached historical data and model artifact loading
src/dashboard_insights.py      Historical summary, calendar, rankings, and evaluation helpers
src/forecast_features.py       Feature construction for saved direct-horizon forecasts
src/live_forecast.py           WAQI forecast readiness and local target-date alignment
src/aqi_alerts.py               Fresh observed-AQI alert rules and session state
src/browser_location.py        Native browser geolocation component interface
src/health_guidance.py         Fresh-observation WAQI-scale public-health guidance
src/waqi_data.py               Read-only recent-observation integration
src/theme.py                   Theme tokens, CSS, and Matplotlib theme application
src/preprocess.py              Raw-to-daily historical data preprocessing
src/features.py                Historical forecast features and target construction
src/train_models.py            Model training and saved evaluation artifacts
data/raw/                      Source historical data
data/processed/                Derived datasets, reports, and evaluation results
models/                        Six saved horizon/model artifacts
tests/                         Unit and Streamlit AppTest coverage
```

## Windows setup

From the project directory in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Configure recent observations (optional)

The WAQI token is read from Streamlit secrets. Create `.streamlit\secrets.toml` locally (it is ignored by Git) with your own token:

```toml
WAQI_API_TOKEN = "YOUR_WAQI_API_TOKEN"
```

Do not commit credentials. Without a configured token or available source, the recent-data workspace reports an unavailable state; it does not substitute historical or sample values.

The CPCB panel uses the data.gov.in resource only when all of the following are configured in Streamlit secrets or the environment: `CPCB_OGD_API_KEY`, `CPCB_OGD_RECORDS_PATH`, `CPCB_OGD_FIELD_MAP` (canonical `location`, `station`, `observed_at`, and `aqi` keys mapped to source fields), and `CPCB_OGD_AQI_IS_OFFICIAL_NATIONAL = true`. The AQI mapping must name a documented direct overall National AQI field (`aqi`, `overall_aqi`, `national_aqi`, or `national_aqi_value`); never map a pollutant average or sub-index to it. Confirm the current resource ID, response schema, units, and meaning of the field against official OGD documentation before enabling the flag. The app matches the configured station exactly and withholds stale, malformed, missing, or mismatched readings. The previous `CPCB_API_KEY` name remains a compatibility fallback; prefer `CPCB_OGD_API_KEY`. If these conditions are not met, CPCB AQI remains unavailable and WAQI continues as a clearly labeled US-scale alternative.

Automatic nearby lookup requires a browser that supports geolocation and a secure context (HTTPS or localhost). Permission is controlled by the browser; denying it leaves the manual city/station workflow available. The WAQI geolocalized feed response must provide station identity and coordinates within 25 km before it is used. Browser coordinates are not written to files or included in application messages; the geolocalized request bypasses the app's response cache.

### Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The six serialized models and processed datasets are already included in the project. Running the pipeline scripts below can replace generated outputs and model files; they are not required to launch the dashboard.

## Forecast methodology and evaluation

Forecasts use the saved direct-horizon Linear Regression and Random Forest models. Historical feature construction uses observations available on or before the forecast origin. The saved models were evaluated with a chronological split: training origins through 2022 and evaluation origins in 2023. Because training horizon targets can extend across the year boundary into 2023, the stored 2023 metrics should not be treated as a leakage-free independent temporal validation. No production model is retrained by the dashboard.

Stored 2023 test-set Random Forest results from `data/processed/model_evaluation_report.txt`:

| Horizon | MAE (dataset AQI units) | RMSE | R² |
|---|---:|---:|---:|
| Day+1 | 6.8817 | 9.1035 | 0.9759 |
| Day+2 | 6.8022 | 8.9625 | 0.9767 |
| Day+3 | 6.8191 | 8.9881 | 0.9766 |

MAE and RMSE are error measures, not accuracy percentages; R² is a coefficient of determination, not an accuracy percentage. These results describe a historical 2023 holdout and do not establish current or future performance. The project also contains expanding-window walk-forward results under `data/processed/walk_forward_results.csv` and its accompanying report. Evaluation numbers shown in the dashboard are read from saved artifacts.

## Data and interpretation limitations

- The historical AQI definition and historical pollutant units have not been independently verified. Do not describe historical values as official CPCB AQI.
- Historical records end on 2023-12-31. Forecast examples targeting January 2024 are historical demonstrations, not current forecasts.
- Recent WAQI values are fetched from the selected, exact-city station feed; when WAQI returns multiple station matches, a station must be selected explicitly. The app uses the direct station `data.aqi` value and preserves its reported value, shows the source observation timestamp separately from retrieval time, and never derives AQI from pollutant values. WAQI success responses are cached by search city or station feed for up to two minutes; failed requests are not cached, and no old successful reading is substituted after a request error.
- WAQI's reported scale is the US EPA 2016 AQI scale documented by the provider at <https://aqicn.org/scale/>. It must not be compared as if it were the Indian CPCB/NAQI scale. Pollutant units are shown only when explicitly provided by the feed.
- The WAQI feed supplies one station observation, not the daily history needed by the saved model's lag and rolling features. Historical dataset AQI provenance and pollutant units/averaging have not been verified against WAQI. Therefore the Recent Forecast page does not emit a same-day estimate or tomorrow/day-after predictions; it explains this limitation and displays dates relative to a valid timestamped station observation when available. WAQI's public data-platform material describes historical data downloads and a separate raw/hourly query process; no supported station-history endpoint is configured in this app. Obtain authorized station-matched history and verify its data contract before developing a separately versioned model.
- Health alerts use only fresh, valid WAQI observations. The threshold AQI ≥101 marks the US EPA 2016 "Unhealthy for Sensitive Groups" band. Alerts are informational and recommend reducing prolonged/strenuous outdoor activity for sensitive groups; they are not diagnosis or personalized medical advice. A cloud overlay runs for about five seconds while a compact dismissible alert remains during the session. Reduced-motion preferences are respected.
- Location rankings depend on the selected historical period and the observed sample count. They should not be generalized beyond that data.
- Health content is informational and does not diagnose, treat, or replace advice from a qualified health professional.
- The saved model metrics do not guarantee forecast quality in a different time period or dataset.

## Tests

Run the requested compile and test checks from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py src tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Screenshots

Add genuine application screenshots here when captured:

- Overview (Dark theme)
- Overview (Light theme)
- Forecast and Model Insights
- Daily AQI Calendar

## Release status

This working folder currently has no Git repository metadata or configured remote. Initialize or open the intended GitHub clone before creating a release; review `.gitignore` and staged files, and never stage `.streamlit\secrets.toml` or environment credentials.
