from .sample_data import get_sample_defects, get_default_case_data, get_pipeline_stages
from .local_db import (
    get_blades,
    get_inspection_options,
    add_blade,
    remove_blade,
    set_blades,
    set_number_of_blades,
    get_include_nacelle,
    set_include_nacelle,
)

__all__ = [
    "get_sample_defects",
    "get_default_case_data",
    "get_pipeline_stages",
    "get_blades",
    "get_inspection_options",
    "add_blade",
    "remove_blade",
    "set_blades",
    "set_number_of_blades",
    "get_include_nacelle",
    "set_include_nacelle",
]
