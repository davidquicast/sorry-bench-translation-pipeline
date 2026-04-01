#!/usr/bin/env python3
"""
Convenience script: translate SorryBench for XSafetyRep.

Usage:
    python scripts/run_sorrybench.py
    python scripts/run_sorrybench.py --tier2-enabled false
    python scripts/run_sorrybench.py --target-languages ar,bn --resume
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Inject default config path and delegate to CLI
sys.argv = [
    "aya-safety", "translate",
    "--config", str(Path(__file__).parent.parent / "configs" / "sorry-bench-202503.yaml"),
    *sys.argv[1:],
]

from aya_safety.cli import main
main()
