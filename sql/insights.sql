-- Core SQL insights
SELECT * FROM vw_daily_kpis;
SELECT * FROM vw_weekday_performance;
SELECT * FROM vw_hourly_activity;
SELECT * FROM vw_top_step_days;
SELECT * FROM vw_low_activity_days;
SELECT * FROM vw_sleep_summary;
SELECT * FROM vw_sleep_vs_activity;

-- Participant rankings
SELECT Id, ActiveDays, AvgSteps, AvgCalories
FROM participant_summary ORDER BY AvgSteps DESC LIMIT 10;

SELECT Id, ActiveDays, AvgSedentaryMinutes
FROM participant_summary ORDER BY AvgSedentaryMinutes DESC LIMIT 10;

-- Daily trend
SELECT Date, ROUND(AVG(TotalSteps),2) avg_steps, ROUND(AVG(Calories),2) avg_calories
FROM daily_activity_clean GROUP BY Date ORDER BY Date;
