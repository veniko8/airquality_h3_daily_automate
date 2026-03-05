# scripts/run_interpolation.py
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def main() -> int:
    # 1) Fail fast if secrets are missing (helps GitHub Actions debugging)
    required = ["SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"Missing environment variables: {missing}", file=sys.stderr)
        return 2

    print("Starting interpolation job…")
    print("Run time (UTC):", datetime.now(timezone.utc).isoformat())

    # 2) Call the daily interpolation pipeline
    try:
        from src.daily_interpolation import main as run_interpolation
        run_interpolation()
        print("\n✅ Interpolation job completed successfully.")
        return 0
    except Exception as e:
        print(f"❌ Interpolation job failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())