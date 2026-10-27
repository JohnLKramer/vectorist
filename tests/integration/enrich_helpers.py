import asyncio
import time

import httpx


async def wait_for_enrichment(client: httpx.AsyncClient, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        counts = (await client.get("/v1/enrichment")).json()
        if counts["pending"] == 0 and counts["running"] == 0:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(f"enrichment did not finish within {timeout}s: {counts}")
        await asyncio.sleep(0.05)
