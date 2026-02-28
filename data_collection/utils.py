from datetime import datetime, timedelta
import csv
import os
import pandas as pd

from openelectricity import OEClient
from openelectricity.types import MarketMetric

from openelectricity.models.timeseries import TimeSeriesResponse


def write_row(csv_path: str, row: dict):
    """Append a single dictionary row to a CSV file, writing headers if file is empty."""
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        # Write header only if file is new or empty
        if not file_exists or os.stat(csv_path).st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def convert_response_to_pandas(response: TimeSeriesResponse) -> pd.DataFrame:
    # TODO: monitor to make sure can handle queries generally
    data = []

    for timeseries in response.data:

        for result in timeseries.results:
            region = result.name.split("_")[-1] 
            for data_point in result.data:
                data.append({
                    "region": region,
                    "timestamp": data_point.timestamp,
                    "metric": timeseries.metric,
                    "value": data_point.value,
                    "unit": timeseries.unit
                })

    return pd.DataFrame(data)


def query_price_by_region(
    region: str,
    interval: str,
    date_start: datetime,
    date_end: datetime
) -> pd.DataFrame:
    """
    Query prices for a region and interval between two datetimes (single API call).
    """

    # Convert to naive datetimes for the API (API expects no tz info in these calls)
    call_start = date_start.replace(tzinfo=None) if date_start.tzinfo is not None else date_start
    call_end = date_end.replace(tzinfo=None) if date_end.tzinfo is not None else date_end

    with OEClient() as client:
        response = client.get_market(
            network_code="NEM",
            metrics=[MarketMetric.PRICE],
            interval=interval,
            date_start=call_start,
            date_end=call_end,
            network_region=region,
        )

    return convert_response_to_pandas(response)

def batch_query_prices(
    region: str,
    interval: str,
    date_start: datetime,
    date_end: datetime
) -> pd.DataFrame:
    """
    Query prices in batches so that no single API call exceeds max_days.
    """
    # Basic validation: require naive datetimes (no tzinfo) because the upstream API
    # expects datetimes without timezone offsets and can return 400 on tz-aware inputs.
    if not isinstance(date_start, datetime) or not isinstance(date_end, datetime):
        raise TypeError(
            "`date_start` and `date_end` must be datetime objects. "
            "If you have strings, parse them with `datetime.fromisoformat()` (no timezone offset)."
        )

    if date_start.tzinfo is not None or date_end.tzinfo is not None:
        raise ValueError(
            "Datetime arguments must be naive (no timezone). The API returns 400 for timezone-aware datetimes. "
            "Provide values like '2026-01-01' or '2026-01-01T00:00:00' (no '+11:00')."
        )

    if date_end < date_start:
        raise ValueError("`date_end` must be the same or after `date_start`.")

    max_days = 32
    all_chunks = []
    cur_start = date_start

    while cur_start <= date_end:
        cur_end = min(cur_start + timedelta(days=max_days - 1), date_end)
        # Use ISO strings in logs so timezone info is clear
        print(f"Fetching: {cur_start.isoformat()} to {cur_end.isoformat()}")

        # Adjust end by 1 second to avoid API inclusive/exclusive boundary errors
        call_end = cur_end - timedelta(seconds=1) if cur_end > cur_start else cur_end
        df = query_price_by_region(region=region, interval=interval, date_start=cur_start, date_end=call_end)
        all_chunks.append(df)

        # Advance to the next instant after cur_end to avoid overlap
        cur_start = cur_end + timedelta(seconds=1)

    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

