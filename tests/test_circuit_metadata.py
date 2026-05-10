"""Sanity checks on the circuit metadata mapping."""

from src.circuit_metadata import CIRCUIT_FAMILIES, get_similar_circuits


def test_no_circuit_lists_itself():
    """A circuit's similar-list should not contain itself."""
    for circuit, similar in CIRCUIT_FAMILIES.items():
        assert circuit not in similar, (
            f"{circuit} appears in its own similar-circuits list"
        )


def test_get_similar_circuits_known():
    """Known circuit returns the expected list."""
    miami_similar = get_similar_circuits("Miami")
    assert isinstance(miami_similar, list)
    assert len(miami_similar) > 0
    assert "Jeddah" in miami_similar


def test_get_similar_circuits_unknown_returns_empty():
    """Unknown circuit returns empty list, not raises."""
    assert get_similar_circuits("Nürburgring") == []
    assert get_similar_circuits("") == []


def test_all_similar_lists_non_empty():
    """If we bother to define a family, it should not be empty."""
    for circuit, similar in CIRCUIT_FAMILIES.items():
        assert len(similar) > 0, f"{circuit} has empty similar list"


def test_no_duplicate_entries_in_lists():
    """Each similar list should have unique entries."""
    for circuit, similar in CIRCUIT_FAMILIES.items():
        assert len(similar) == len(set(similar)), (
            f"{circuit} has duplicates in its similar list"
        )
