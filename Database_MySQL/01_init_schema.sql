/******************************************************************************************
    PROJECT TITLE:   Predictive Marketing & Economic Analysis DB
AUTHOR:          Davy GOUPIL
DATE:            2026-02-11
CONTEXT:         RNCP Title "AI Developer" - Block 1 Certification
DESCRIPTION:     
    This database consolidates client CRM data, marketing campaign logs, 
    and ECB macroeconomic indicators to optimize term deposit subscriptions.
    It utilizes a Star Schema architecture for efficient KPI reporting.
******************************************************************************************/

/* ---------------------------------------------------------
   1. CREATE DATABASE
--------------------------------------------------------- */
CREATE DATABASE IF NOT EXISTS bank_marketing_dw;
USE bank_marketing_dw;


/* ---------------------------------------------------------
  DROP TABLE 
--------------------------------------------------------- */

DROP TABLE IF EXISTS dim_client;
DROP TABLE IF EXISTS dim_campaign;
DROP TABLE IF EXISTS dim_economics;
DROP TABLE IF EXISTS fact_interactions;


/* ---------------------------------------------------------
   1. CREATE TABLES
--------------------------------------------------------- */

-- 1. CLIENT DIMENSION
-- Simple storage of client demographics.
CREATE TABLE IF NOT EXISTS `dim_client` (
    `client_id` INTEGER NOT NULL AUTO_INCREMENT,
    `client_age` INTEGER NOT NULL,
    `job_category` VARCHAR(50),
    `marital_status` VARCHAR(20),
    `education_level` VARCHAR(50),
    `has_credit_default` TINYINT(1) NOT NULL DEFAULT 0, 
    `has_housing_loan` TINYINT(1) NOT NULL DEFAULT 0,
    `has_personal_loan` TINYINT(1) NOT NULL DEFAULT 0,
    `account_balance` DECIMAL(10,2) NOT NULL,          
    PRIMARY KEY (`client_id`)
);

-- 2. CAMPAIGN DIMENSION
-- Stores the marketing context.
CREATE TABLE IF NOT EXISTS `dim_campaign` (
    `campaign_id` INTEGER NOT NULL AUTO_INCREMENT,
    `campaign_name` VARCHAR(100) NOT NULL,            
    `campaign_start_date` DATE NOT NULL,
    `campaign_end_date` DATE NOT NULL,
    `campaign_channel` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`campaign_id`)
);

-- 3. ECONOMICS DIMENSION
-- Stores the macro-economic context.
CREATE TABLE IF NOT EXISTS `dim_economics` (
    `economics_id` INTEGER NOT NULL,                 
    `economics_date` DATE NOT NULL UNIQUE,
    `ecb_rate` DECIMAL(4,2) NOT NULL,                 
    `rate_description` VARCHAR(255) NOT NULL,
    `currency` CHAR(3) NOT NULL DEFAULT 'EUR',
    PRIMARY KEY (`economics_id`)
);

-- 4. FACT TABLE (INTERACTIONS)
-- The central table linking everything.
CREATE TABLE IF NOT EXISTS `fact_interactions` (
    `interaction_id` INTEGER NOT NULL AUTO_INCREMENT,
    `contact_type` VARCHAR(20) NOT NULL,
    `last_contact_day` INTEGER NOT NULL,
    `last_contact_month` VARCHAR(10) NOT NULL,
    `contact_duration_sec` INTEGER NOT NULL,
    `contacts_this_campaign` INTEGER NOT NULL,
    `days_since_previous_contact` INTEGER,              -- NULL if never contacted
    `nb_previous_interactions` INTEGER NOT NULL DEFAULT 0,
    `prev_campaign_outcome` VARCHAR(20),
    `has_subscribed_target` TINYINT(1) NOT NULL DEFAULT 0,
    
    -- Foreign Keys columns
    `client_id` INTEGER NOT NULL,
    `campaign_id` INTEGER NOT NULL,
    `economics_id` INTEGER NOT NULL,
    
    PRIMARY KEY (`interaction_id`)
);

-- 5. FOREIGN KEYS
-- Link Interaction -> Client
ALTER TABLE `fact_interactions`
ADD CONSTRAINT `fk_client`
FOREIGN KEY (`client_id`) REFERENCES `dim_client`(`client_id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;

-- Link Interaction -> Campaign
ALTER TABLE `fact_interactions`
ADD CONSTRAINT `fk_campaign`
FOREIGN KEY (`campaign_id`) REFERENCES `dim_campaign`(`campaign_id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;

-- Link Interaction -> Economics
ALTER TABLE `fact_interactions`
ADD CONSTRAINT `fk_economics`
FOREIGN KEY (`economics_id`) REFERENCES `dim_economics`(`economics_id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;