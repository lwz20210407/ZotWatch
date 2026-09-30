"""Run unit tests with live HTTP disabled, even if a developer has real API keys."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


if __name__ == '__main__':
    with patch('requests.sessions.Session.request', side_effect=AssertionError(
            'Unit tests cannot use live HTTP; mock the request or replay cached responses')):
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
