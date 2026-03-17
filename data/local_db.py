"""Local database for blade configuration and inspection inputs."""
import json
import os

# Same folder as this package (wind_turbine_inspection_app/data/)
DB_DIR = os.path.dirname(os.path.abspath(__file__))
BLADES_DB_PATH = os.path.join(DB_DIR, "blades_db.json")

DEFAULT_BLADES = [
    {"id": "361", "name": "Blade 361"},
    {"id": "363", "name": "Blade 363"},
    {"id": "379", "name": "Blade 379"},
]
NACELLE_OPTION = {"id": "nacelle", "name": "Nacelle"}


def _load_raw():
    if os.path.isfile(BLADES_DB_PATH):
        with open(BLADES_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"blades": DEFAULT_BLADES.copy(), "include_nacelle": True}


def _save_raw(data):
    with open(BLADES_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_blades():
    """Return list of blade dicts: [{"id": "361", "name": "Blade 361"}, ...]."""
    return _load_raw().get("blades", DEFAULT_BLADES.copy())


def get_include_nacelle():
    return _load_raw().get("include_nacelle", True)


def set_include_nacelle(include):
    data = _load_raw()
    data["include_nacelle"] = bool(include)
    _save_raw(data)


def set_blades(blades, include_nacelle=True):
    """Replace full blades list and nacelle flag."""
    data = _load_raw()
    data["blades"] = list(blades)
    data["include_nacelle"] = bool(include_nacelle)
    _save_raw(data)


def add_blade(blade_id, name=None):
    """Append a blade to the list. name defaults to 'Blade {id}'."""
    data = _load_raw()
    blades = list(data.get("blades", []))
    blade_id = str(blade_id).strip()
    name = (name or f"Blade {blade_id}").strip()
    if not blade_id:
        return blades
    # Avoid duplicate id
    blades = [b for b in blades if b.get("id") != blade_id]
    blades.append({"id": blade_id, "name": name})
    data["blades"] = blades
    _save_raw(data)
    return data["blades"]


def remove_blade(blade_id):
    """Remove a blade by id."""
    data = _load_raw()
    blades = [b for b in data.get("blades", []) if b.get("id") != str(blade_id)]
    data["blades"] = blades
    _save_raw(data)
    return data["blades"]


def set_number_of_blades(n):
    """Ensure at least n blades exist; if more, trim to n (by order)."""
    n = max(0, int(n))
    data = _load_raw()
    blades = list(data.get("blades", []))
    if len(blades) < n:
        for i in range(len(blades), n):
            blades.append({"id": str(400 + i), "name": f"Blade {400 + i}"})
    else:
        blades = blades[:n]
    data["blades"] = blades
    _save_raw(data)
    return data["blades"]


def get_inspection_options():
    """Return list of options for inspection dropdown: Blade 361, Blade 363, Nacelle, etc."""
    data = _load_raw()
    blades = data.get("blades", DEFAULT_BLADES.copy())
    options = [b.get("name") or f"Blade {b.get('id', '')}" for b in blades]
    if data.get("include_nacelle", True):
        options.append(NACELLE_OPTION["name"])
    return options
