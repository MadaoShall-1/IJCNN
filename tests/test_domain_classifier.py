import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.domain_classifier import classify_domain


def test_electricity_domain():
    domains, subdomains, confidence = classify_domain("A capacitor is connected to a voltage source.")
    assert "electricity" in domains
    assert "capacitors" in subdomains
    assert confidence >= 0.5


def test_waves_domain():
    domains, subdomains, _ = classify_domain("A wave has frequency and wavelength.")
    assert domains == ["waves"]
    assert "wave_relation" in subdomains
