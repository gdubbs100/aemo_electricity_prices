from openelectricity.types import DataMetric
from datetime import datetime
from openelectricity import OEClient

import aiohttp
import os
from aiohttp.resolver import ThreadedResolver

# Patch aiohttp to use the system resolver (same as requests)
aiohttp.connector.DefaultResolver = ThreadedResolver

from dotenv import load_dotenv
load_dotenv()  # This looks for the .env file and loads it into os.environ

if __name__ == "__main__":
    # Initialize with environment variables (recommended)
    client = OEClient()
    print(f'Key found: {os.environ.get("OPENELECTRICITY_API_KEY")[:10]}...')

    response = client.get_network_data(
        network_code="NEM",
        metrics=[DataMetric.POWER, DataMetric.ENERGY],
        interval="1h",
        date_start=datetime(2024, 1, 1),
        date_end=datetime(2024, 1, 2),
        primary_grouping="network_region",
        secondary_grouping="fueltech"
    )

    # Access data
    for timeseries in response.data:
        print(f"Metric: {timeseries.metric}")
        for result in timeseries.results:
            for data_point in result.data:
                print(f"  {data_point.timestamp}: {data_point.value}")
