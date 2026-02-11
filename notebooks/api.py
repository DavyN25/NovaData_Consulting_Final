import os
from typing import List, Optional
from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymysql
from dotenv import load_dotenv

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------
# We initialize FastAPI with metadata that will appear in the auto-generated /docs
app = FastAPI(
    title="Bank Marketing Data API",
    description="Professional API exposing Client, Campaign, and KPI data stored in MySQL",
    version="1.0.0"
)

# --------------------------------------------------
# DATABASE CONNECTION LOGIC
# --------------------------------------------------
def get_db_connection():
    """
    Establishes a connection to the MySQL database using environment variables.
    Uses DictCursor to return results as dictionaries (perfect for JSON/API).
    """
    load_dotenv()  # Securely load credentials from a .env file

    try:
        return pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306)),
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        # If DB connection fails, return a 500 error to the client
        raise HTTPException(status_code=500, detail=f"Database Connection Error: {str(e)}")

# --------------------------------------------------
# DATA MODELS (Pydantic)
# --------------------------------------------------
# These models define the "contract" of our API. 
# They ensure data validation and clear documentation for the jury.

class Client(BaseModel):
    client_id: int
    job_category: Optional[str]
    account_balance: float
    has_housing_loan: bool

class Campaign(BaseModel):
    campaign_name: str
    campaign_channel: str
    start_date: date

class KPI(BaseModel):
    total_interactions: int
    total_sales: int
    conversion_rate: float

# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/", tags=["General"])
def root():
    """Landing page to check API status."""
    return {"status": "online", "architecture": "FastAPI + PyMySQL", "docs": "/docs"}


@app.get("/clients", response_model=List[Client], tags=["Data Access"])
def get_clients(skip: int = 0, limit: int = 10, job: Optional[str] = None):
    """
    Retrieves a list of clients with Pagination (skip/limit).
    Optionally filters by job category.
    """
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # Dynamic SQL construction based on filters
            if job:
                sql = "SELECT client_id, job_category, account_balance, has_housing_loan FROM dim_client WHERE job_category = %s LIMIT %s OFFSET %s"
                cursor.execute(sql, (job, limit, skip))
            else:
                sql = "SELECT client_id, job_category, account_balance, has_housing_loan FROM dim_client LIMIT %s OFFSET %s"
                cursor.execute(sql, (limit, skip))

            result = cursor.fetchall()
            return result
    finally:
        connection.close()  # Always close the connection to prevent memory leaks


@app.get("/campaigns", response_model=List[Campaign], tags=["Data Access"])
def get_campaigns():
    """Retrieves all marketing campaigns from the dimension table."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT campaign_name, campaign_channel, campaign_start_date AS start_date FROM dim_campaign"
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        connection.close()


@app.get("/kpi/conversion", response_model=KPI, tags=["Analytics"])
def get_conversion_kpi():
    """
    Aggregates real-time KPIs from the fact table.
    Calculates total interactions, successes, and the percentage rate.
    """
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT 
                COUNT(*) AS total_interactions, 
                SUM(has_subscribed_target) AS total_sales,
                ROUND(SUM(has_subscribed_target) / COUNT(*) * 100, 2) AS conversion_rate
            FROM fact_interactions
            """
            cursor.execute(sql)
            return cursor.fetchone()
    finally:
        connection.close()