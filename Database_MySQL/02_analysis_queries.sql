/* ---------------------------------------------------------
   1. SQL INSIGHT QUERIES
--------------------------------------------------------- */


-- -----------------------------------------------------
-- Query 1 — Subscription Rate by Campaign Channel
-- -----------------------------------------------------
SELECT 
    c.campaign_channel,
    AVG(f.has_subscribed_target) AS subscription_rate
FROM FACT_INTERACTIONS f
JOIN DIM_CAMPAIGN c ON f.campaign_id = c.campaign_id
GROUP BY c.campaign_channel;

-- -----------------------------------------------------
-- Query 2 — Impact of Interest Rates on Subscription
-- -----------------------------------------------------
SELECT 
    e.ecb_rate,
    AVG(f.has_subscribed_target) AS subscription_rate
FROM FACT_INTERACTIONS f
JOIN DIM_ECONOMICS e ON f.economics_date = e.economics_date
GROUP BY e.ecb_rate
ORDER BY e.ecb_rate;

-- -----------------------------------------------------
-- Query 3 — Client Balance Sensitivity
-- -----------------------------------------------------
SELECT 
    CASE 
        WHEN cl.account_balance < 0 THEN 'Negative'
        WHEN cl.account_balance < 5000 THEN 'Low'
        ELSE 'High'
    END AS balance_segment,
    AVG(f.has_subscribed_target) AS conversion_rate
FROM FACT_INTERACTIONS f
JOIN DIM_CLIENT cl ON f.client_id = cl.client_id
GROUP BY balance_segment;


-- -----------------------------------------------------
-- Query 4 — Campaign Effectiveness by Economic Context
-- -----------------------------------------------------
SELECT 
    c.campaign_channel,
    AVG(e.ecb_rate) AS avg_rate,
    AVG(f.has_subscribed_target) AS conversion
FROM FACT_INTERACTIONS f
JOIN DIM_CAMPAIGN c ON f.campaign_id = c.campaign_id
JOIN DIM_ECONOMICS e ON f.economics_date = e.economics_date
GROUP BY c.campaign_channel;


-- -----------------------------------------------------
-- Query 2 — Impact of Interest Rates on Subscription
-- -----------------------------------------------------