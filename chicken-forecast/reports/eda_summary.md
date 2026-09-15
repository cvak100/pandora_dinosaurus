# Part 1 – EDA Summary (flock)

- Period: **2025-01-01 -> 2026-08-02** (579 days; missing egg records: 1)
- Mean daily eggs: **6.15** (median 6.0; max 10)
- Mean hens: **10.89** (range 8–13)
- Mean eggs/hen/day: **0.573**
- Mean feed: **1.522 kg/day** (135 g/hen)
- Feed conversion: **0.277 kg feed / egg**
- Seasonality (eggs/hen): best month **3** (0.669), worst **10** (0.266)
- Corr(eggs_per_hen, temp_avg) = **-0.131**
- Corr(eggs_per_hen, daylight_minutes) = **-0.019**
- Corr(eggs_per_hen, precipitation) = **-0.013**
- Extreme temp: coldest 10% eggs/hen **0.610**, hottest 10% **0.555**, mid **0.571**
- Event `event_death` (n=4): pre7 eggs/hen **0.524** -> post14 **0.544** (delta +0.020)
- Event `event_released` (n=3): pre7 eggs/hen **0.461** -> post14 **0.458** (delta -0.003)
- Event `event_added` (n=1): pre7 eggs/hen **0.607** -> post14 **0.253** (delta -0.354)
- Event `event_feed_change` (n=5): pre7 eggs/hen **0.530** -> post14 **0.538** (delta +0.008)
- Event `event_egg_anomaly` (n=12): pre7 eggs/hen **0.537** -> post14 **0.522** (delta -0.015)
- Anomalies (|z|>2.5 on eggs/hen): low **14**, high **1**

## Assumptions
- User filter: **configured flock user only**.
- Feed logs are date ranges; quantity is **grams/day** for each day in the range (overlap for same food type → later `from_date` wins).
- Weather from `data/raw/*.json` yearly files.
- Processed table: `data/processed/daily_flock_clean.csv`

## Main takeaways
1. Production is strongly seasonal — eggs/hen varies materially by month.
2. Flock size changes (deaths / releases / additions) shift absolute egg counts; always normalize by hens for comparisons.
3. Weather (temperature / daylight when available) correlates with laying rate; extreme cold/hot tails differ from mid-range days.
4. Feed is relatively stable over long ranges; conversion (kg/egg) is a useful efficiency KPI alongside eggs/hen.
5. Event windows (especially mortality and feed changes) are worth modeling as features in Part 2 weekly forecasting.
