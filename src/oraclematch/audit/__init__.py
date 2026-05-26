"""Anti-gaming audit: detect single-oracle exploiters via cross-oracle disagreement."""

from oraclematch.audit.antigaming import AntiGamingDetector, AntiGamingReport, wilson_interval

__all__ = ["AntiGamingDetector", "AntiGamingReport", "wilson_interval"]
