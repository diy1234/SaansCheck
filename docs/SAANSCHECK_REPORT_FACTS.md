# SaansCheck — Academic Report Facts

**Final project title:** SaansCheck — Delhi-NCR Air Quality Intelligence and Personalized Risk Advisor  
**Project:** MCA Minor Project, Semester 3  
**Audit status:** Evidence-based facts are separated below from unknowns and limitations. This is a technical fact sheet, not a claim that every limitation has been resolved.

## 1. Final project title

SaansCheck — Delhi-NCR Air Quality Intelligence and Personalized Risk Advisor.

## 2. Abstract facts

**VERIFIED FACTS**

- SaansCheck is a Streamlit application with separate Historical Intelligence and Recent Air Quality workspaces.
- Historical screens explore a supplied Delhi-NCR dataset and display stored model evaluations and saved-model forecasts.
- The recent-observation workflow uses WAQI station feeds, explicit station identity, source timestamps, retrieval time, freshness validation, and US EPA 2016 AQI category guidance.
- The application has a separate CPCB National AQI path. It fails closed when a directly documented overall-AQI field cannot be verified or retrieved; it does not convert WAQI values into CPCB values.
- Health guidance and alerts are informational. They are not medical diagnoses.

**NOT ESTABLISHED**

- The historical AQI field's upstream source, generating formula, and national scale.
- Whether historical pollutant records represent individual physical monitoring stations or city-level data.
- Whether the historical pollutant values have the units implied by some display conventions.

## 3. Problem statement

Air-quality information is distributed across historical data, live station observations, forecasts, and public-health context. The project provides one interface for exploring those distinct information types while labelling source, time, scale, and uncertainty. It must not present different AQI standards or historical predictions as interchangeable live observations.

## 4. Project objectives

1. Explore historical Delhi-NCR air-quality records and trends.
2. Compare the five dataset location labels over selected periods.
3. Demonstrate saved, multi-horizon AQI regression models and their recorded evaluation results.
4. Present recent provider observations with station identity, source timestamp, freshness, and source scale.
5. Provide cautious, scale-specific general health information and observed-data alerts.
6. Clearly report unavailable data rather than fabricating a reading.

## 5. Scope

**Included:** Historical visualizations, location and pollutant analysis, historical calendar/seasonal/health context, saved-model insights and forecasts, WAQI station search/feed, browser geolocation where supported, CPCB unavailable-state handling, themes, and navigation.

**Excluded or unavailable:** Verified overall live CPCB AQI; AQI.in automatic retrieval; a validated current WAQI-to-historical-model bridge; a live future forecast from recent observations; and independently established provenance for the historical AQI target.

## 6. Technologies and libraries

The inspected `requirements.txt` lists Python packages `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, and `joblib`. Source also imports `requests` for provider HTTP access. The UI uses Streamlit and scoped CSS/HTML; the browser-location component uses browser geolocation. Tests use Python `unittest` and Streamlit `AppTest`.

The application environment inspected for this audit used Python 3.10.11 and Streamlit 1.65.0. Dependency versions are not pinned in `requirements.txt`.

## 7. Hardware and software requirements

**VERIFIED SOFTWARE:** Python 3.10.11 environment, Streamlit 1.65.0 in the inspected environment, and the dependencies listed above. The project runs on Windows in the inspected workspace; automatic browser geolocation requires a secure context (HTTPS or localhost) and user permission.

**NOT AVAILABLE:** No measured CPU, memory, storage, or minimum hardware specification is documented. No deployment/server sizing benchmark was performed.

## 8. Dataset details and provenance

| Dataset artifact | Shape | Date coverage | Purpose |
|---|---:|---|---|
| `data/raw/delhi_ncr_air_quality.csv.csv` | 219,125 rows x 18 columns | Hourly timestamps 2019-01-01 00:00 through 2024-01-01 00:00 | Supplied raw historical records |
| `data/processed/daily_air_quality.csv` | 9,130 rows x 23 columns | 2019-01-01 through 2023-12-31 | Complete-day aggregated data |
| `data/processed/ml_ready_dataset.csv` | 9,115 rows x 31 columns | 2019-01-04 through 2023-12-31 | Lag/rolling features and direct-horizon targets |

Five dataset location labels are present: Delhi, Faridabad, Ghaziabad, Gurgaon, and Noida. The raw CSV contains `timestamp`, `location_name`, `aqi`, `aqi_category`, pollutant fields (`co`, `no2`, `o3`, `pm10`, `pm25`, `so2`), and location metadata. It has no upstream source identifier, station ID, AQI-method/version field, or pollutant-unit columns. The audited raw file has no missing cells; its final 2024-01-01 date is incomplete.

**Provenance.** The investigation note `data/processed/kaggle_provenance_investigation.txt` records a public Kaggle dataset page by uploader `aniket0712`, describing a Delhi-NCR hourly dataset and labelling it CC0. The Kaggle listing calls its file `processed_aqi_data.csv`; the local file's identity was not established by downloading and hashing the Kaggle artifact. The source does not identify an upstream monitoring agency, collection process, AQI formula, or named AQI standard. Accordingly, local AQI is described as **dataset-defined historical AQI**, not verified CPCB AQI.

## 9. Data preprocessing

`src/preprocess.py` parses and sorts hourly timestamps by location, checks missing/duplicate/negative values, removes calendar dates with fewer than 24 hourly records, and calculates arithmetic daily means for the six pollutant fields and supplied `aqi`. The partial 2024-01-01 day is excluded, leaving 9,130 location-days through 2023-12-31. The script derives calendar fields and category labels using CPCB numeric breakpoints, but it does **not** calculate AQI from pollutant concentrations. There is no imputation in the observed raw-to-daily path; the raw file was reported complete.

## 10. Exploratory data analysis

The Historical Intelligence workspace provides historical summaries, temporal/pollutant visualizations, location comparison/rankings, a daily AQI calendar, seasonal analysis, health context, and model-insight views. The inspected overview showed the latest historical date as 2023-12-31 and exposed summary statistics for the selected period/location. These are exploratory views of the supplied dataset, not validation of its source or AQI method.

## 11. Feature engineering

`src/features.py` constructs direct targets `AQI_t+1`, `AQI_t+2`, and `AQI_t+3` by forward-shifting `aqi` **within each location group**. It removes the initial three lag-boundary records per location and creates five one-hot location features. The 22 predictors are:

- Historical AQI: `aqi`, `AQI_lag_1`, `AQI_lag_2`, `AQI_lag_3`, `AQI_rolling_mean_3`, `AQI_rolling_mean_7`, `AQI_rolling_std_7`.
- Dataset pollutant fields: `pm25`, `pm10`, `no2`, `so2`, `co`, `o3`.
- Calendar: `month`, `day`, `day_of_week`, `is_weekend`.
- Location: `loc_Delhi`, `loc_Faridabad`, `loc_Ghaziabad`, `loc_Gurgaon`, `loc_Noida`.

Lag and rolling features are backward-looking. Units for pollutant features remain undocumented; the field names do not prove physical units.

## 12. Machine-learning methodology

Two regressors are trained for each direct forecast horizon. Training uses rows whose **feature/origin year** is through 2022 and evaluation uses 2023 origins, with no row shuffling. The Linear Regression model is scikit-learn ordinary least squares. The Random Forest uses 100 estimators, maximum depth 15, minimum split 4, minimum leaf 2, `random_state=42`, and `n_jobs=-1`.

**Temporal-boundary caveat:** targets are shifted within location, but the main training partition is selected using feature-row year. Training rows at the end of 2022 can therefore have 2023 target dates, overlapping the nominal 2023 evaluation period. The stored 2023 holdout metrics are not a leakage-free temporal evaluation. The repository also contains target-date-aware expanding-window validation results described in section 14.

## 13. Models and saved artifacts

Six existing `.joblib` artifacts were inspected and preserved:

- `models/linear_regression_day1.joblib`
- `models/linear_regression_day2.joblib`
- `models/linear_regression_day3.joblib`
- `models/random_forest_day1.joblib`
- `models/random_forest_day2.joblib`
- `models/random_forest_day3.joblib`

They represent Linear Regression and Random Forest models for Day+1, Day+2, and Day+3. The dashboard loads these artifacts; it does not retrain them on application launch.

## 14. Evaluation metrics and results

**Stored 2023 origin-year holdout — Random Forest** (`data/processed/model_evaluation_report.txt`):

| Horizon | MAE | RMSE | R² |
|---|---:|---:|---:|
| Day+1 | 6.8817 | 9.1035 | 0.9759 |
| Day+2 | 6.8022 | 8.9625 | 0.9767 |
| Day+3 | 6.8191 | 8.9881 | 0.9766 |

These are dataset AQI units. The temporal-boundary caveat in section 12 applies. They are not current performance guarantees. R² is a coefficient of determination, **not an accuracy percentage**.

**Target-date-aware expanding-window validation — pooled Random Forest means across three windows** (`data/processed/walk_forward_validation_report.txt`):

| Horizon | MAE mean | RMSE mean | R² mean |
|---|---:|---:|---:|
| Day+1 | 6.828 | 8.995 | 0.9764 |
| Day+2 | 6.802 | 8.944 | 0.9767 |
| Day+3 | 6.895 | 9.042 | 0.9762 |

The windows validate target dates in 2021, 2022, and 2023, with training target dates capped before each validation window. This is the more appropriate leakage-aware result in the repository. Metrics still depend on the undocumented AQI target and dataset coverage.

**Derived historical category evaluation (Model Insights):** `data/processed/test_predictions.csv` provides paired actual and predicted AQI values for both regressors and all three horizons. The dashboard applies the same thresholds as `src/preprocess.py` to both sides; this is a category evaluation derived from regression outputs, not a separately trained classifier. Fixed label order is Good, Satisfactory, Moderate, Poor, Very Poor, Severe; sklearn metrics use `zero_division=0`, retaining unsupported classes in macro averages. For the stored 2023 forecast-origin rows:

| Model | Horizon | Scored n | Excluded rows | Macro P | Macro R | Macro F1 | Weighted P | Weighted R | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear Regression | Day+1 | 1,820 | 5 | 0.459087 | 0.460540 | 0.459783 | 0.922701 | 0.922527 | 0.922567 |
| Linear Regression | Day+2 | 1,815 | 10 | 0.458807 | 0.459730 | 0.459254 | 0.922028 | 0.921763 | 0.921872 |
| Linear Regression | Day+3 | 1,810 | 15 | 0.452415 | 0.452651 | 0.452522 | 0.909252 | 0.908840 | 0.909021 |
| Random Forest | Day+1 | 1,820 | 5 | 0.463076 | 0.463045 | 0.463041 | 0.929557 | 0.929121 | 0.929295 |
| Random Forest | Day+2 | 1,815 | 10 | 0.462250 | 0.461931 | 0.462054 | 0.927903 | 0.927273 | 0.927504 |
| Random Forest | Day+3 | 1,810 | 15 | 0.459208 | 0.458539 | 0.458778 | 0.921650 | 0.920442 | 0.920829 |

Actual support in the fixed label order is `Good, Satisfactory, Moderate, Poor, Very Poor, Severe`: Day+1 `(0, 421, 677, 722, 0, 0)`, Day+2 `(0, 421, 672, 722, 0, 0)`, Day+3 `(0, 421, 667, 722, 0, 0)`. Good, Very Poor, and Severe have zero test support; their zero-valued class metrics affect the six-class macro average. The UI shows the model/horizon confusion matrix and per-class precision, recall, F1, actual support, and predicted count. The exact implemented intervals are <=50, <=100, <=200, <=300, <=400, and >400. Although preprocessing calls these CPCB categories, the upstream historical AQI source/calculation/scale provenance is unverified; do not report these as verified CPCB performance. Scores also inherit the 2022-training/2023-target boundary overlap above.

## 15. Streamlit application architecture

`app.py` owns application configuration, workspace/navigation state, page rendering, theme integration, and composition of the recent and historical views. `src/data_loader.py` validates and caches datasets, evaluation artifacts, and the six saved models. Historical preprocessing/training scripts are separate from normal dashboard use. Live WAQI and CPCB integrations are independent paths.

## 16. Module-by-module explanation

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit entry point, navigation, workspace/page rendering, recent and historical integration |
| `src/data_loader.py` | Cached/schema-checked CSV and `.joblib` loading |
| `src/preprocess.py` | Hourly-to-daily historical aggregation and backward features |
| `src/features.py` | Training dataset predictors and direct future targets |
| `src/train_models.py` | Offline model training and stored evaluation exports |
| `src/forecast_features.py` | Inference-time feature contract and direct horizon forecasts |
| `src/dashboard_insights.py` | Historical summary, ranking, calendar, and evaluation helpers |
| `src/walk_forward_validation.py` | Target-date-aware expanding-window validation |
| `src/waqi_data.py` | WAQI station search, feed requests, validation, freshness, and source metadata |
| `src/browser_location.py` | Browser geolocation component integration |
| `src/cpcb_national_aqi.py` | Fail-closed data.gov.in CPCB National AQI lookup |
| `src/health_guidance.py` | WAQI US-scale guidance for valid fresh observations |
| `src/aqi_alerts.py` | Fresh observed-AQI alert construction, session state, and transient overlay |
| `src/aqi_risk_scale.py` | Scale-specific category bands/current-marker data for the risk UI |
| `src/live_forecast.py` | Recent-data forecast availability/readiness status |
| `src/theme.py` | Dark/light theme tokens, scoped CSS, chart theme |
| `src/ui_components.py` | Reusable dashboard components |
| `src/aqi_calculator.py` | CPCB-style pollutant/AQI calculations used by separate validation/utility paths; not the historical CSV's AQI provenance |
| `src/aqi_validation.py` | Research-only comparison against assumed CPCB-style calculations |
| `src/conformal_intervals.py` | Forecast interval utilities and stored validation support |
| `src/live_data.py`, `src/xkdr_data.py` | Present in source tree; no active `app.py` route was established in this audit |

## 17. Historical Intelligence workflow

The dashboard loads `daily_air_quality.csv`, model artifacts, and saved evaluation outputs. Historical pages filter or group this dataset for overview, pollutant/trend analysis, location comparison, AQI calendar, seasonal context, health context, and model insights. The Forecast view builds a 22-column origin row from a selected dataset location and calls each saved direct-horizon model. It uses no forecast output as another horizon's input.

The latest complete historical observation is 2023-12-31. The inspected forecast page showed January 1, 2, and 3, 2024 target dates. These are historical demonstrations, **not future forecasts relative to today**.

## 18. Recent WAQI observation workflow

The user can permit browser geolocation or use manual city/station selection and locality search. Automatic lookup is accepted only with returned station identity and coordinates within 25 km. Manual search uses provider station results, deduplicates by station UID, and requires explicit station selection when multiple matches exist. The selected WAQI station feed supplies the direct provider `data.aqi` value; the application does not compute AQI from pollutant values. The observation timestamp, retrieval time, source scale, and freshness are separate. Missing, stale, malformed, or mismatched feeds produce an unavailable state rather than a substituted reading.

**Browser observation captured during this audit:** Anand Vihar, Delhi; WAQI-reported AQI 269 on WAQI's US EPA 2016 scale; provider timestamp `2026-11-09 22:00+05:30`; application freshness status FRESH at capture. This is a time-specific browser observation, not a persistent project constant or a present-day guarantee. The provider-reported station label was `Anand Vihar, Delhi, Delhi, India`.

## 19. CPCB integration status

Indian National AQI — CPCB is the preferred Indian standard in the interface. The configured official data.gov.in resource ID is `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`. Previous bounded connectivity diagnosis found DNS, TCP 443, and public HTTPS available, but the authenticated resource request timed out. The current overall CPCB value therefore remains unavailable. The UI displays the unavailable state and links to the official CPCB dashboard: <https://airquality.cpcb.gov.in/AQI/>.

An API key alone is insufficient to promote a response as overall National AQI. A documented field map, matching response schema, exact station, observation time, and proof that the mapped field is an official overall AQI are required. Pollutant averages/sub-indices are not overall AQI. Do not enable the official-AQI configuration flag without this evidence.

## 20. AQI health advisory and alert logic

WAQI publishes the US EPA 2016 scale used by the app: Good 0–50, Moderate 51–100, Unhealthy for Sensitive Groups 101–150, Unhealthy 151–200, Very Unhealthy 201–300, and Hazardous 301+. Health guidance is shown only when there is a valid, fresh, timestamped WAQI observation. High-AQI alerts require a fresh identified observation at AQI 101 or above; predictions and stale data do not trigger alerts.

CPCB categories are separate: Good 0–50, Satisfactory 51–100, Moderate 101–200, Poor 201–300, Very Poor 301–400, Severe 401–500. WAQI values are never classified with CPCB thresholds. Health text is general public-health information, not diagnosis or treatment advice.

## 21. Animated risk-scale UI

The Recent Air Quality landing page has a responsive six-band AQI risk scale with distinct CPCB and WAQI category bands, iconography, gradient, and a live marker only when a valid observation for the selected scale is available. The unavailable CPCB state has no CPCB current marker. The UI uses scoped styling and reduced-motion handling; it does not require continuous reruns for animation. During browser verification the risk-scale section rendered in the recent workspace. The captured image is viewport-cropped and includes an active high-AQI alert overlay; it is genuine app output, not a composited graphic.

## 22. Functional test results

**Browser observations verified during this audit or prior audit steps:** Recent Air Quality / Recent Overview rendered; the manual WAQI locality search returned a matching Anand Vihar station; selecting it rendered the real provider-reported AQI 269, station name, timestamp, FRESH status, US-scale category guidance, and high-AQI alert; CPCB remained unavailable with its official link. Historical Overview, Location Intelligence, AQI Forecast, Model Insights, Health Advisor, Daily AQI Calendar, Seasonal Risk, and Learn About AQI were visited without a Streamlit exception. Pollution Analytics rendered a pollutant statistics table; its page-heading text check was inconclusive. The Recent Pollutant Trends, Location Data, Data Freshness, and Forecast routes were not individually verified in this final browser pass.

| Check | Result | Evidence/qualification |
|---|---|---|
| Historical Intelligence workspace | PASS | Historical pages visited; overview and forecasts rendered |
| Recent Air Quality workspace | PASS | Real selected-station observation and no-data states rendered |
| Navigation pages | PARTIAL | Listed historical routes and Recent Overview rendered; remaining recent routes were not individually verified, and Pollution Analytics heading check was inconclusive |
| Historical AQI trends/visualizations | PASS | Historical analytics rendered in browser |
| Location intelligence/city comparison | PASS | Page and comparison view rendered |
| Day+1/Day+2/Day+3 historical forecasts | PASS | Forecast page rendered dates 2024-01-01 through 2024-01-03 |
| Linear Regression/Random Forest insights | PASS | Saved evaluation/insight page rendered |
| Health advisor and alerts | PASS | Health page and AQI 269 alert were observed |
| WAQI station selection/locality search | PASS | Anand Vihar result and feed selected; result is source/availability dependent |
| Animated AQI risk scale | PASS | Recent landing page scale rendered; no CPCB current marker while CPCB unavailable |
| Dark and Light themes | INCOMPLETE | Dark theme was visibly rendered. The Light control is present, but browser-panel constraints prevented a reliable Light-theme visual verification. |
| CPCB unavailable state/link | PASS | Unavailable message and official link rendered |

**Automated validation:** `python -m compileall -q app.py src tests` completed successfully. `python -m unittest discover -s tests -v` completed successfully: **153 tests passed, 0 failures/errors**. A separate HTTP check of `http://127.0.0.1:8501/` returned **200**. Streamlit emitted expected bare-mode cache/ScriptRunContext warnings during unit tests.

## 23. Screenshots and evidence paths

All listed PNGs are located under `docs/report_evidence/` and were captured from the running Streamlit application. The embedded browser constrained the viewport and repeatedly placed a real high-AQI alert over content. Therefore **do not treat every file as a clean, print-ready image**. The following images are useful as captured evidence: `04_aqi_observation_and_health_alert.png` (real Anand Vihar observation and alert), `05_historical_overview.png`, `06_historical_pollution_trends.png`, `07_city_comparison.png`, `08_historical_aqi_forecast.png`, `09_model_insights.png`, and `10_health_advisor.png`. The other listed screenshots need manual recapture at presentation resolution, without clipping or overlays.

1. `docs/report_evidence/01_home_recent_air_quality.png`
2. `docs/report_evidence/02_animated_aqi_risk_scale.png`
3. `docs/report_evidence/03_waqi_station_search.png`
4. `docs/report_evidence/04_aqi_observation_and_health_alert.png`
5. `docs/report_evidence/05_historical_overview.png`
6. `docs/report_evidence/06_historical_pollution_trends.png`
7. `docs/report_evidence/07_city_comparison.png`
8. `docs/report_evidence/08_historical_aqi_forecast.png`
9. `docs/report_evidence/09_model_insights.png`
10. `docs/report_evidence/10_health_advisor.png`
11. `docs/report_evidence/11_cpcb_unavailable_state.png`

**Manual recapture recommended:** 01, 02, 03, and 11. In addition, inspect framing of 05–10 before printing. Capture 01 at the top of Recent Air Quality; capture 02 with the full scale visible and no alert overlay; capture 03 with the manual query and actual station results visible; capture 11 with the CPCB unavailable message and official link readable. The AQI 269 and timestamp shown in 04 are a browser capture from the audit time, not static application data.

## 24. Verified project achievements

- Keeps historical dataset AQI, WAQI US-scale observations, and model predictions as separate data types.
- Requires station identity/validation for recent observation display and exposes a freshness status.
- Provides a graceful CPCB-unavailable state without re-labelling WAQI as CPCB.
- Trains and stores three horizons for two algorithms; the dashboard loads saved artifacts.
- Has station-isolated target shifts and target-date-aware walk-forward validation artifacts.
- Provides source-scale-specific public-health content and observation-based alert rules.

## 25. Limitations and future scope

**Current limitations:** historical source and AQI formula are unverified; pollutant units are not locally documented; local city labels are not verified station identities; authenticated CPCB OGD access timed out; no live AQI.in retrieval is implemented; recent WAQI feeds do not supply the matched historical data contract required by the saved model; historical demo forecast dates are in 2024; and the stored origin-year 2023 holdout has target-date overlap.

**Reasonable future work (not performed in this audit):** obtain and document source-level historical provenance and units; verify the CPCB data contract and a genuine overall-AQI field; acquire permissioned station-matched history before adding live-feed forecasts; add a data dictionary and versioned evaluation protocol; and recapture clean, full-page screenshots at presentation resolution.

## 26. References and dataset URLs

- Kaggle dataset page documented in the provenance investigation: <https://www.kaggle.com/datasets/aniket0712/delhi-ncr-hourly-air-quality-dataset-20192024>
- Kaggle metadata endpoint documented in the provenance investigation: <https://www.kaggle.com/api/v1/datasets/view/aniket0712/delhi-ncr-hourly-air-quality-dataset-20192024>
- Official CPCB National AQI dashboard: <https://airquality.cpcb.gov.in/AQI/>
- data.gov.in CPCB resource ID used by the application: `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`
- WAQI/AQICN scale documentation: <https://aqicn.org/scale/>
- Detailed local provenance notes: `data/processed/kaggle_provenance_investigation.txt`, `data/processed/aqi_provenance_investigation.txt`
- Model evaluation source artifacts: `data/processed/model_evaluation_report.txt`, `data/processed/walk_forward_validation_report.txt`
