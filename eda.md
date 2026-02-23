## Phase I: Data Quality — Findings & Decisions

Data quality was assessed across 12 candidate cities over the full 2-year period
(May 2023 – April 2025). Three filtering criteria were applied at the sensor level:
maximum gap duration > 30 days, 75th percentile of gap distribution > 10 days,
and more than 20% of months with daily coverage below 70%. This reduced the dataset
from 12 to 6 cities (Berlin, London, Lyon, New York, Paris, Rome), eliminating
Barcelona, Delhi, Los Angeles, and Santiago due to insufficient or unstable data.
Aberrant values (negative PM2.5) were deleted rather than imputed to preserve gap
detection integrity. A systematic coverage drop was observed across European sensors
in March–May 2024, attributed to an upstream OpenAQ reporting issue rather than
sensor degradation — this period is flagged but sensors were not penalized for it.
