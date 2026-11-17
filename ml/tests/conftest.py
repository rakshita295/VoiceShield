"""
Pytest configuration for VoiceShield ML subsystem.
Ensures that the 'ml' root directory is present in sys.path.
"""

import sys
from pathlib import Path

# Add ml/ directory to sys.path so 'import src...' resolves cleanly
ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
