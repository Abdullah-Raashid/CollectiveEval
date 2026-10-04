"""Generate Phase 8 reports offline; does not run inference."""

from pathlib import Path

from collectiveeval.pilot_analysis import analyze

if __name__ == "__main__":
    result = analyze(Path("reports/pilot_v1"))
    print(result["provider_accounting"]["combined"])
