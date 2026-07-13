"""
scheduler.py - Entry point for Render Cron Jobs.

Calls data_refresh.run_refresh() with the FRED API key from environment.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_refresh import run_refresh


def main():
    api_key = os.environ.get('FRED_API_KEY', '')
    if not api_key:
        print('ERROR: FRED_API_KEY environment variable is required')
        sys.exit(1)
    print(f'Scheduler starting with FRED API key: {api_key[:8]}...')
    results = run_refresh(api_key=api_key)
    
    # Print summary
    print('\n--- Summary ---')
    for source, info in results.items():
        status = info.get('status', 'unknown')
        if status == 'success':
            print(f'  {source}: OK ({info.get("rows", "?")} rows, last: {info.get("last_date", "?")})')
        elif status == 'error':
            print(f'  {source}: ERROR - {info.get("message", "unknown")}')
        else:
            print(f'  {source}: {status}')
    
    # Exit with error if any source failed
    failed = [k for k, v in results.items() if v.get('status') == 'error']
    if failed:
        print(f'\nWARNING: {len(failed)} source(s) failed: {", ".join(failed)}')
        sys.exit(1)
    
    print('\nAll sources refreshed successfully.')


if __name__ == '__main__':
    main()
