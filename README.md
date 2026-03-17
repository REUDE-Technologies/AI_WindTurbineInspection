# Wind Turbine Inspection AI Dashboard (Streamlit)

Dashboard for AI-powered wind turbine inspection from drone aerial videos: defect categorization and instant report generation.

## Run the app

```bash
cd wind_turbine_inspection_app
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Features

- **Key metrics:** Total defects, Critical/High count, Nacelle vs Blades, coverage %, AI confidence
- **Pipeline status:** Ingest → Segmentation → Detection → Classification → Report ready
- **Charts:** Defects by severity (pie), by type (bar), by component/view, by blade zone
- **Defect table:** Filter by severity, component, type; sort by severity; expand for detail
- **Visual explorer:** Drill down by Nacelle view (Top/Bottom/Back/Left/Right/Front) or by Blade (361, 379, 363) and side (SS/PS)
- **Case data:** Edit report metadata in the sidebar (location, WTG, team, drone, weather)
- **Instant report:** Generate report preview and download as HTML or Markdown

## Data

Sample data is included and matches the structure of the REUDE_ABANPower Kayathar inspection report (nacelle views, blades 361/379/363, severity classes, defect types and recommendations).
