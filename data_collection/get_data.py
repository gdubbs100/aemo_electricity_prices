import aiohttp
from aiohttp.resolver import ThreadedResolver
# Patch aiohttp to use the system resolver (same as requests)
aiohttp.connector.DefaultResolver = ThreadedResolver

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from utils import batch_query_prices

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Download NEM price data for a region and date range.")
    parser.add_argument("--region", type=str, required=True, help="NEM region code (e.g. VIC1)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    parser.add_argument("--interval", type=str, default="1h", help="Interval (default: 1h)")
    parser.add_argument("--tz", type=str, default="Australia/Melbourne", help="Timezone (default: Australia/Melbourne)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV file path")
    args = parser.parse_args()

    tz = ZoneInfo(args.tz)

    def parse_and_validate(dt_str: str) -> datetime:
        """Parse ISO date or datetime string and validate it is naive (no timezone).

        Raises ValueError with instructions if parsing fails or a timezone is present.
        Acceptable inputs:
        - YYYY-MM-DD
        - YYYY-MM-DDTHH:MM:SS
        (Do NOT include a timezone offset.)
        """
        try:
            dt = datetime.fromisoformat(dt_str)
        except Exception:
            raise ValueError(
                f"Invalid date format: '{dt_str}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS' without timezone offset."
            )

        if dt.tzinfo is not None:
            raise ValueError(
                f"Datetime '{dt_str}' includes a timezone offset. The API expects naive datetimes (no timezone). "
                "Provide values like '2026-01-01' or '2026-01-01T00:00:00' instead."
            )

        return dt

    start_dt = parse_and_validate(args.start)
    end_dt = parse_and_validate(args.end)

    print(f"Querying {args.region} from {start_dt} to {end_dt} (interval={args.interval})")
    df = batch_query_prices(args.region, args.interval, start_dt, end_dt)

    if args.out:
        out_path = f"data/{args.out}.csv"
    else:
        start_str = start_dt.strftime("%Y-%m-%d_%H-%M-%S")
        end_str = end_dt.strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"data/{args.region}_{start_str}_to_{end_str}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()


