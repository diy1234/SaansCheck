"""Observed-only health alerts for WAQI's documented AQI scale."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, MutableMapping


HIGH_AQI_THRESHOLD = 101
HIGH_AQI_SEVERITY = "Unhealthy for Sensitive Groups"
HIGH_AQI_PRECAUTION = (
    "Sensitive groups should consider reducing prolonged or strenuous outdoor activity."
)


def build_observed_aqi_alert(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Create an alert only for a fresh, identified provider observation."""
    aqi = observation.get("aqi")
    observed_at = observation.get("observed_at")
    station_id = observation.get("station_id")
    station_name = observation.get("station_name")
    if (
        not observation.get("available")
        or observation.get("freshness") != "FRESH"
        or isinstance(aqi, bool)
        or not isinstance(aqi, (int, float))
        or not math.isfinite(float(aqi))
        or float(aqi) < HIGH_AQI_THRESHOLD
        or not isinstance(observed_at, datetime)
        or observed_at.tzinfo is None
        or station_id is None
        or not isinstance(station_name, str)
        or not station_name.strip()
    ):
        return None

    timestamp_key = observed_at.isoformat()
    value = float(aqi)
    if value <= 150:
        severity = "Unhealthy for Sensitive Groups"
    elif value <= 200:
        severity = "Unhealthy"
    elif value <= 300:
        severity = "Very Unhealthy"
    else:
        severity = "Hazardous"
    precaution = HIGH_AQI_PRECAUTION
    if severity in {"Unhealthy", "Very Unhealthy"}:
        precaution = (
            "Consider reducing prolonged or strenuous outdoor activity; "
            "sensitive groups should take extra care."
        )
    elif severity == "Hazardous":
        precaution = (
            "Avoid strenuous outdoor activity and consider remaining indoors "
            "when practical."
        )
    event_key = f"{station_id}|{timestamp_key}|{severity}"
    return {
        "event_key": event_key,
        "station_id": str(station_id),
        "station_name": station_name,
        "aqi": value,
        "observed_at": observed_at,
        "severity": severity,
        "precaution": precaution,
    }


def update_alert_session_state(
    session_state: MutableMapping[str, Any],
    alert: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    """Persist one alert event across reruns and identify genuinely new events."""
    event_key = alert["event_key"] if alert else None
    previous_key = session_state.get("saanscheck_aqi_alert_event_key")
    is_new = event_key is not None and event_key != previous_key

    if event_key is None:
        session_state["saanscheck_aqi_alert_event_key"] = None
        session_state.pop("saanscheck_aqi_alert_dismissed_key", None)
        session_state.pop("saanscheck_aqi_alert_data", None)
        return None, False

    if is_new:
        session_state["saanscheck_aqi_alert_event_key"] = event_key
        session_state.pop("saanscheck_aqi_alert_dismissed_key", None)
        session_state["saanscheck_aqi_alert_data"] = alert
    else:
        session_state["saanscheck_aqi_alert_data"] = alert

    if session_state.get("saanscheck_aqi_alert_dismissed_key") == event_key:
        return None, is_new
    return alert, is_new


def dismiss_alert_session_state(session_state: MutableMapping[str, Any]) -> None:
    """Dismiss the current station/timestamp/severity event until it changes."""
    event_key = session_state.get("saanscheck_aqi_alert_event_key")
    if event_key is not None:
        session_state["saanscheck_aqi_alert_dismissed_key"] = event_key


def cloud_alert_component_html(alert: dict[str, Any] | None) -> str:
    """Return a tiny client-side bridge for the five-second cloud overlay.

    The Streamlit component iframe asks the parent page to own the overlay and
    its timer, so ordinary Streamlit reruns do not restart or cut short it.
    """
    cloud = None
    if alert is not None:
        cloud = {
            "event_key": alert["event_key"],
            "station_name": alert["station_name"],
            "aqi": alert["aqi"],
            "observed_label": alert["observed_at"].isoformat(),
            "precaution": alert["precaution"],
        }
    payload = json.dumps(cloud, ensure_ascii=True).replace("<", "\\u003c")
    return f"""<!doctype html>
<html><body><script>
(() => {{
  const alert = {payload};
  let parentWindow;
  let doc;
  try {{
    parentWindow = window.parent;
    doc = parentWindow.document;
  }} catch (error) {{
    console.error("SaansCheck health alert overlay could not access the dashboard.", error);
    return;
  }}
  const nodeId = "saanscheck-waqi-cloud-alert";
  const timerName = "__saanscheckWaqiCloudTimer";
  const existing = doc.getElementById(nodeId);
  if (!alert) {{
    if (parentWindow[timerName]) {{
      parentWindow.clearTimeout(parentWindow[timerName]);
      parentWindow[timerName] = null;
    }}
    if (existing) existing.remove();
    return;
  }}
  if (existing && existing.dataset.eventKey === alert.event_key) return;
  if (parentWindow.__saanscheckWaqiLastCloudEventKey === alert.event_key) return;
  if (parentWindow[timerName]) parentWindow.clearTimeout(parentWindow[timerName]);
  if (existing) existing.remove();

  const styleId = "saanscheck-waqi-cloud-alert-style";
  if (!doc.getElementById(styleId)) {{
    const style = doc.createElement("style");
    style.id = styleId;
    style.textContent = `
      @keyframes sc-waqi-cloud-motion {{
        0% {{ opacity: 0; transform: translateY(16px) scale(.96); }}
        9% {{ opacity: 1; transform: translateY(0) scale(1); }}
        22% {{ transform: translateY(-3px); }}
        36% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-2px); }}
        64% {{ transform: translateY(0); }}
        86% {{ opacity: 1; transform: translateY(0); }}
        100% {{ opacity: 0; transform: translateY(-14px); }}
      }}
      #saanscheck-waqi-cloud-alert {{
        position: fixed; z-index: 2147483000; top: 76px; right: 22px;
        width: min(330px, calc(100vw - 28px)); box-sizing: border-box;
        padding: 17px 20px; border: 1px solid #f59e0b;
        border-radius: 38px 38px 30px 30px;
        background: var(--sc-surface, #111c2e); color: var(--sc-text, #f1f5f9);
        box-shadow: 0 12px 38px #f59e0b55, 0 4px 18px #0005;
        font: 500 14px/1.45 system-ui, sans-serif;
        animation: sc-waqi-cloud-motion 5s ease-in-out both;
        pointer-events: none;
      }}
      #saanscheck-waqi-cloud-alert::before,
      #saanscheck-waqi-cloud-alert::after {{
        content: ""; position: absolute; z-index: -1; border-radius: 50%;
        background: var(--sc-surface, #111c2e); border: 1px solid #f59e0b;
        border-bottom-color: transparent;
      }}
      #saanscheck-waqi-cloud-alert::before {{
        width: 48px; height: 34px; top: -14px; left: 34px;
      }}
      #saanscheck-waqi-cloud-alert::after {{
        width: 62px; height: 42px; top: -20px; right: 42px;
      }}
      #saanscheck-waqi-cloud-alert strong {{
        display: block; color: #fbbf24; font-size: 16px; margin-bottom: 3px;
      }}
      #saanscheck-waqi-cloud-alert .sc-cloud-meta {{
        color: var(--sc-muted, #a8b7ca); font-size: 12px; margin-top: 5px;
      }}
      @media (max-width: 640px) {{
        #saanscheck-waqi-cloud-alert {{ top: 12px; right: 10px; }}
      }}
      @media (prefers-reduced-motion: reduce) {{
        #saanscheck-waqi-cloud-alert {{ animation: none !important; }}
      }}
    `;
    doc.head.appendChild(style);
  }}
  const node = doc.createElement("div");
  node.id = nodeId;
  node.dataset.eventKey = alert.event_key;
  node.setAttribute("role", "alert");
  const title = doc.createElement("strong");
  title.textContent = "⚠ High AQI Alert";
  const detail = doc.createElement("div");
  detail.textContent = `AQI ${{alert.aqi}} · ${{alert.station_name}}`;
  const observed = doc.createElement("div");
  observed.className = "sc-cloud-meta";
  observed.textContent = `Observed ${{alert.observed_label}}`;
  const guidance = doc.createElement("div");
  guidance.className = "sc-cloud-meta";
  guidance.textContent = alert.precaution;
  node.append(title, detail, observed, guidance);
  doc.body.appendChild(node);
  parentWindow.__saanscheckWaqiLastCloudEventKey = alert.event_key;
  parentWindow[timerName] = parentWindow.setTimeout(() => {{
    const current = doc.getElementById(nodeId);
    if (current && current.dataset.eventKey === alert.event_key) current.remove();
    parentWindow[timerName] = null;
  }}, 5000);
}})();
</script></body></html>"""
