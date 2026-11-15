# Deviations from prereg/2026-11-14_forecast_audit.md

Kept outside the frozen file so its hash stays valid. Each entry says what changed, when, and
what had been seen.

1. **First full run failed, 2026-11-14 about 05:31 to 05:38 UTC.** 7,120 of 7,182 calls returned
   "api_key must be set"; the key variable did not reach the script. No model was called and
   nothing was cached. The run was restarted at about 06:13 with the key exported.
2. **glm-5.2 dropped, 2026-11-14 about 06:20 UTC.** In the restarted run, 17 of its first 25 new
   calls used the full 16,000 output tokens on reasoning and returned no answer. The frozen rule
   drops a model above 10% failed calls. It was applied early, on those 25 calls, to stop
   spending about 0.07 credits per empty answer. Nothing about forecast values was looked at:
   the decision used only token counts and empty outputs. The 11 glm-5.2 answers that did
   parse (pilot and restart) stay in the cache and are excluded from every analysis. The run
   continues with six models via `--models`; no frozen file changed.
3. **Leakage probe spend, 05:33 UTC.** The first 110 probe calls added 0.163 credits to the main
   spend counter (data/spend.json). No forecast or cache entry was affected.
4. **Sub-claim run, 2026-11-14.** Three models exceeded the 10% failed-call cutoff on the
   sub-claim set (deepseek-v4-flash 25.9%, glm-5.3-flash 28.4%, qwen3.8-27b 44.8%; empty answers
   at the token cap plus 75 provider errors). Per the frozen rule they are dropped from the
   Amendment 1 claims and shown descriptively. Three models remain.
5. **Erratum in the frozen main file.** It says catalogue outcomes were fixed "between
   2026-10-15 and 2026-11-12". The `updated` dates in results/commons.json run from 2026-10-15
   to 2026-11-08. The frozen file is left as is; the report uses the correct dates.
6. **Post-freeze analysis scripts.** analyze_h.py, analyze_a.py and make_figures.py were written
   after the runs. They apply the registered rules. Two choices were not registered and were
   fixed before computing: "pooled" is the equal-weight mean of the models' item forecasts, and
   H3/A3 ("skill at or below zero") count as supported only if the interval's upper end is <= 0.
