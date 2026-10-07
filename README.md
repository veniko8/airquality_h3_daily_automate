# airquality_h3_daily_automate

# Zagreb Air Quality & Environmental Exposure Dashboard 🚲🌳

An automated, end-to-end data pipeline and visualization system that monitors real-time air quality and weather conditions across Zagreb, designed specifically to help parents evaluate pollution exposure around local kindergartens and outdoor spaces.

---

## 🚀 Project Overview

The project automates the extraction, transformation, storage, and visualization of environmental metrics across Zagreb. It captures official station data and meteorological conditions, stores them securely in a cloud database, runs wind-aware geospatial interpolation, and presents storytelling insights via executive dashboards, dynamic hex maps, and automated AI narratives.

---

## 🛠️ Tech Stack Architecture

The pipeline consists of four core legs:

1. **Automation & Ingestion (n8n):** Self-hosted workflows handling scheduled hourly API pulls from official environmental endpoints (`iszz.azo.hr`) and Open-Meteo, automated daily backfills, error-alerting protocols, and LLM-driven daily summaries.
2. **Backend & Storage (Supabase):** Cloud PostgreSQL database managing relational time-series tables (`airquality`, `weather_hourly`), schema migrations, timezone-corrected timestamps (`timestamptz`), and analytical views (`gold_aqi_weather_hourly`, `daily_aqi_summary`).
3. **Analytics & BI (Power BI):** Interactive executive dashboards featuring multi-pollutant trends ($PM_{2.5}$, $PM_{10}$, $NO_2$), time-series heatmaps, automated AI text block integration, and dynamic kindergarten ranking layers.
4. **Geospatial Processing (VS Code + Kepler.gl):** Python script execution leveraging Uber's H3 spatial indexing and wind-aware Inverse Distance Weighting (IDW) interpolation to map pollution diffusion across Zagreb, visualized interactively through Kepler.gl.

---

## ⚙️ Workflow - Pipeline & Storytelling Focus

* **Data Ingestion & Backfilling:** n8n fetches hourly data from official monitoring endpoints and Open-Meteo, using robust upsert operations and automated error monitoring to ensure data integrity.
* **Data Transformation & Storage:** Raw metrics are cleaned, standardized to correct timestamps, and stored in Supabase tables feeding downstream views.
* **Spatial Modeling:** Python scripts pull station data and wind parameters from Supabase, run daily H3 hexagonal interpolations, and output formatted datasets for mapping.
* **Kindergarten Exposure Storytelling:** The system couples official Zagreb kindergarten locations with spatial data, allowing parents to assess how bad air quality is at different kindergartens generally, by time of day, and by day of the week via Power BI matrices and spatial layers.
* **Reporting & AI Summaries:** Automated daily text summaries generated via LLM integrations inside n8n are piped directly into Power BI text visuals to give instant readability alongside deep telemetry.