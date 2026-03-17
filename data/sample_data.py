"""Sample inspection data aligned with REUDE_ABANPower Kayathar report structure."""
import pandas as pd
from datetime import date

def get_sample_defects():
    return pd.DataFrame([
        # Nacelle Top
        {"defect_id": "N-Top-D1", "component": "Nacelle", "view_zone": "Top", "blade_id": "", "side": "", "zone": "", "condition": "Large irregular patch with exposed area", "defect_type": "Coating damage / paint peel", "severity": "High", "recommendation": "Schedule clean, recoat", "confidence": 0.92},
        {"defect_id": "N-Top-D2", "component": "Nacelle", "view_zone": "Top", "blade_id": "", "side": "", "zone": "", "condition": "Small spot near edge", "defect_type": "Minor local stain", "severity": "Low", "recommendation": "Monitor re-check", "confidence": 0.88},
        {"defect_id": "N-Top-D3", "component": "Nacelle", "view_zone": "Top", "blade_id": "", "side": "", "zone": "", "condition": "Long area with surface streaks", "defect_type": "Surface contamination / dust", "severity": "Low", "recommendation": "Clean for clearer defect confirmation", "confidence": 0.85},
        {"defect_id": "N-Top-D4", "component": "Nacelle", "view_zone": "Top", "blade_id": "", "side": "", "zone": "", "condition": "Sealant edge rusting", "defect_type": "Sealant / coating wear", "severity": "Medium", "recommendation": "Close-up inspection", "confidence": 0.90},
        {"defect_id": "N-Top-D5", "component": "Nacelle", "view_zone": "Top", "blade_id": "", "side": "", "zone": "", "condition": "Patchy dirt/abrasion", "defect_type": "Local contamination", "severity": "Low", "recommendation": "Clean and re-check", "confidence": 0.82},
        # Nacelle Bottom
        {"defect_id": "N-Bot-D1", "component": "Nacelle", "view_zone": "Bottom", "blade_id": "", "side": "", "zone": "", "condition": "Large dark spot with surface discoloration", "defect_type": "Local contamination / impact stain", "severity": "Medium", "recommendation": "Clean area and schedule coating assessment", "confidence": 0.87},
        {"defect_id": "N-Bot-D4", "component": "Nacelle", "view_zone": "Bottom", "blade_id": "", "side": "", "zone": "", "condition": "Vertical streak from edge region", "defect_type": "Sealant leak trace / runoff stain", "severity": "High", "recommendation": "Priority check / inspect nearby joints/seals", "confidence": 0.91},
        # Nacelle Back
        {"defect_id": "N-Back-D3", "component": "Nacelle", "view_zone": "Back", "blade_id": "", "side": "", "zone": "", "condition": "Left hardware strip near fastener", "defect_type": "Corrosion / wear", "severity": "Medium", "recommendation": "Close-up inspect of strip, fastener corrosion", "confidence": 0.89},
        {"defect_id": "N-Back-D5", "component": "Nacelle", "view_zone": "Back", "blade_id": "", "side": "", "zone": "", "condition": "Brownish edge band", "defect_type": "Dirt, rust runoff staining", "severity": "Medium", "recommendation": "Corrective coating", "confidence": 0.84},
        # Blade 361 SS
        {"defect_id": "B361-SS-D1", "component": "Blade", "view_zone": "Root", "blade_id": "361", "side": "SS", "zone": "Root (0-15%)", "condition": "Small dark spot", "defect_type": "Coating chip / grease or oil", "severity": "Low", "recommendation": "Inspect and clean", "confidence": 0.86},
        {"defect_id": "B361-SS-D2", "component": "Blade", "view_zone": "Root-Midspan", "blade_id": "361", "side": "SS", "zone": "Root to Midspan (15-50%)", "condition": "Narrow hairline at trailing edge", "defect_type": "Scratch / coating line", "severity": "Medium", "recommendation": "Inspect opening edges and schedule crack repair", "confidence": 0.88},
        {"defect_id": "B361-SS-D3", "component": "Blade", "view_zone": "Midspan-Tip", "blade_id": "361", "side": "SS", "zone": "Midspan to Tip (50-80%)", "condition": "Dark patch with splatter pattern", "defect_type": "Bird strike / impact stain", "severity": "Medium", "recommendation": "Close-up inspection, monitor recurrence", "confidence": 0.90},
        {"defect_id": "B361-SS-D4", "component": "Blade", "view_zone": "Tip", "blade_id": "361", "side": "SS", "zone": "Tip (80-100%)", "condition": "Darkened band along tip leading edge", "defect_type": "Leading-edge erosion / tip coating wear", "severity": "Medium", "recommendation": "Close-up inspection to confirm erosion", "confidence": 0.87},
        # Blade 361 PS
        {"defect_id": "B361-PS-D1", "component": "Blade", "view_zone": "Root", "blade_id": "361", "side": "PS", "zone": "Root (0-25%)", "condition": "Minor localized stain", "defect_type": "Grease/oil staining", "severity": "Low", "recommendation": "Clean and re-image", "confidence": 0.83},
        # Blade 379
        {"defect_id": "B379-SS-D1", "component": "Blade", "view_zone": "Tip", "blade_id": "379", "side": "SS", "zone": "Tip (75-100%)", "condition": "Localized surface mark near tip edge", "defect_type": "Contamination / coating wear", "severity": "Low", "recommendation": "Close-up re-imaging at higher zoom", "confidence": 0.85},
        {"defect_id": "B379-PS-D1", "component": "Blade", "view_zone": "Root-Midspan", "blade_id": "379", "side": "PS", "zone": "Root to Midspan (0-30%)", "condition": "Localized stain near root", "defect_type": "Grease/oil residue", "severity": "Low", "recommendation": "Clean and repeat imaging", "confidence": 0.81},
        # Blade 363
        {"defect_id": "B363-SS-D1", "component": "Blade", "view_zone": "Tip", "blade_id": "363", "side": "SS", "zone": "Tip (100%)", "condition": "Localized mark near tip edge", "defect_type": "Contamination / coating wear stain", "severity": "Low", "recommendation": "Close-up re-imaging", "confidence": 0.84},
        {"defect_id": "B363-PS-D1", "component": "Blade", "view_zone": "Root-Midspan", "blade_id": "363", "side": "PS", "zone": "Root to Midspan (30-70%)", "condition": "Long linear indication", "defect_type": "Stain or early edge crack / bond line", "severity": "Medium", "recommendation": "Verification priority at higher zoom", "confidence": 0.89},
    ])

def get_default_case_data():
    return {
        "report_number": "REU_WT_IR_001_2026",
        "report_date": date.today().isoformat(),
        "location": "Wind Turbine Test Station (WTTS), Kayathar, Tamil Nadu",
        "wtg_type": "Vestas RRB 225 kW",
        "blade_serials": "361, 379, 363",
        "field_work_start": "2025-12-29",
        "field_work_end": "2025-12-30",
        "operator": "Arunachalam Karunaivel",
        "reviewer": "Rajesh Kumar Balaram",
        "verifier": "Reginald Chellam",
        "drone_model": "REUDE MINI 850",
        "software": "REUDE Compliq",
        "weather": "Sunny",
        "wind_ms": "4 - 6",
        "temp_c": "34 - 38",
    }

def get_pipeline_stages():
    return [
        {"name": "Video/Image ingest", "status": "done", "eta": ""},
        {"name": "Blade/Nacelle segmentation", "status": "done", "eta": ""},
        {"name": "Defect detection", "status": "done", "eta": ""},
        {"name": "Defect classification", "status": "done", "eta": ""},
        {"name": "Report data ready", "status": "done", "eta": ""},
    ]
