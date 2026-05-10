"""
Circuit metadata: maps each F1 circuit to a list of stylistically similar tracks.

Used by feature engineering to compute `driver_similar_circuits_avg` — the
driver's average finish at tracks that share key characteristics with the
target track (street vs permanent, downforce level, overtaking difficulty).

Replacing the hardcoded Miami-only list with a per-circuit mapping is what
generalises the pipeline to predict any race, not just Miami.

The groupings are based on:
- Layout type (street circuit / hybrid / permanent)
- Average overtaking opportunities (low / medium / high)
- Downforce level (low / medium / high)
- Tyre degradation profile

Sources: F1 technical previews, Pirelli tyre allocation reports.
"""

# Groupings indexed by canonical circuit name (the `Circuit` field from FastF1).
# When a circuit is not in this dict, the feature falls back to a season-wide
# expanding average (handled in feature_engineering.py).
CIRCUIT_FAMILIES: dict[str, list[str]] = {
    # Street / hybrid circuits with limited overtaking — qualifying matters
    "Miami":         ["Jeddah", "Las Vegas", "Baku", "Melbourne", "Singapore"],
    "Monaco":        ["Singapore", "Baku", "Jeddah"],
    "Singapore":     ["Monaco", "Baku", "Jeddah", "Miami"],
    "Baku":          ["Jeddah", "Las Vegas", "Miami", "Monaco"],
    "Jeddah":        ["Miami", "Las Vegas", "Baku", "Singapore"],
    "Las Vegas":     ["Jeddah", "Miami", "Baku"],
    "Melbourne":     ["Miami", "Montréal", "Imola"],

    # High-downforce permanent circuits with technical sectors
    "Barcelona":     ["Hungaroring", "Suzuka", "Zandvoort"],
    "Hungaroring":   ["Monaco", "Barcelona", "Singapore"],
    "Suzuka":        ["Silverstone", "Spa-Francorchamps", "Barcelona"],
    "Zandvoort":     ["Hungaroring", "Barcelona"],

    # Power circuits — long straights, low downforce
    "Monza":         ["Spa-Francorchamps", "Baku"],
    "Spa-Francorchamps": ["Monza", "Silverstone", "Suzuka"],
    "Silverstone":   ["Spa-Francorchamps", "Suzuka", "Austin"],
    "Austin":        ["Silverstone", "Mexico City", "São Paulo"],

    # Mid-downforce semi-permanent
    "Imola":         ["Melbourne", "Montréal", "Barcelona"],
    "Montréal":      ["Imola", "Baku", "Melbourne"],
    "Mexico City":   ["Austin", "São Paulo"],
    "São Paulo":     ["Austin", "Mexico City"],

    # Middle-east / desert permanents
    "Sakhir":        ["Yas Island", "Lusail"],   # Bahrain
    "Yas Island":    ["Sakhir", "Lusail"],       # Abu Dhabi
    "Lusail":        ["Yas Island", "Sakhir"],   # Qatar

    # Asia-Pacific / others
    "Shanghai":      ["Sakhir", "Yas Island"],
}


def get_similar_circuits(circuit: str) -> list[str]:
    """Return the list of stylistically similar circuits for a given track.

    If the circuit is unknown, returns an empty list — the feature engineering
    code will fall back to the driver's overall recent form in that case.
    """
    return CIRCUIT_FAMILIES.get(circuit, [])
