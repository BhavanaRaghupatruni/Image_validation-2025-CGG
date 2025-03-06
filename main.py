import os
import psycopg2
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Form
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import tempfile
from typing import List
from fastapi.security import HTTPBasic
from validation import validate_image

# Setting up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()          # Log to console
    ]
)

logger = logging.getLogger(__name__)

security = HTTPBasic()
app = FastAPI()

# PostgreSQL connection details
DB_HOST = "10.2.28.48"
DB_PORT = 6399
DB_NAME = "imagevalidation"
DB_USER = "imagevalidation"  
DB_PASSWORD = "!Mag@vali#25"  

# Function to authenticate the user
def authenticate(username: str = Form(...), password: str = Form(...)):
    valid_username = "admin"
    valid_password = "password"
    if username != valid_username or password != valid_password:
        logger.warning(f"Authentication failed for username: {username}")
        raise HTTPException(status_code=401, detail="Invalid login")  # Changed from 'message' to 'detail'
    logger.info(f"User '{username}' authenticated successfully")
    return username

# Function to get system IP address
def get_system_ip_address(request: Request):
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        ip_addresses = x_forwarded_for.split(',')
        if ip_addresses:
            return ip_addresses[0].strip()
    return request.client.host

# Function to connect to the PostgreSQL database
def connect_db():
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        logger.info("Database connection established successfully.")
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")  # Changed from 'message' to 'detail'

# Function to insert validation results into the database
def insert_validation_result(filename: str, status: str, reason: str, user: str, ip_address: str):
    connection = connect_db()
    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO result_table (filename, validity, reason, "User", timestamp, ip_address)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (filename, status, reason, user, datetime.now(), ip_address))
        connection.commit()
        logger.info(f"Validation result inserted into database for file: {filename}")
    except Exception as e:
        logger.error(f"Error inserting into the database for {filename}: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()

@app.post("/validate-image/")
async def validate_image_endpoint(username: str = Depends(authenticate), file: UploadFile = File(...), request: Request = None):
    logger.info(f"User '{username}' uploaded file: {file.filename}, content type: {file.content_type}")

    if not file.content_type.startswith("image/"):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")  # Changed from 'message' to 'detail'

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
            logger.info(f"Temporary file created at {temp_file_path}")

        result = validate_image(temp_file_path, min_size_kb=4, max_size_kb=100, min_fullscan=8.0, max_fullscan=100.0)

    except Exception as e:
        logger.exception(f"Error during image validation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")  # Changed from 'message' to 'detail'
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} removed.")

    # Handle result based on validation outcome
    if result.get("status") == "success":
        logger.info(f"Image validation successful: {file.filename}")
        insert_validation_result(file.filename, "Valid", "", username, get_system_ip_address(request))
        return JSONResponse(content=result)

    error_reason = result.get("reason", "Validation failed due to an unknown issue.")
    logger.warning(f"Image validation failed: {error_reason}")

    insert_validation_result(file.filename, "Invalid", error_reason, username, get_system_ip_address(request))
    raise HTTPException(status_code=400, detail=error_reason)  # Changed from 'message' to 'detail'

@app.post("/validate-images/") 
async def validate_images(request: Request, username: str = Depends(authenticate), files: List[UploadFile] = File(...)):
    client_ip = get_system_ip_address(request)
    logger.info(f"Validation request received from IP: {client_ip}")

    for file in files:
        logger.info(f"Processing file: {file.filename} (Content Type: {file.content_type})")

        if not file.content_type.startswith("image/"):
            logger.warning(f"Invalid file type: {file.filename} ({file.content_type})")
            insert_validation_result(file.filename, "Invalid", "Invalid file type.", username, client_ip)
            continue

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name
                logger.info(f"Saved temporary file: {temp_file_path}")

            result = validate_image(temp_file_path, min_size_kb=4, max_size_kb=100, min_fullscan=8.0, max_fullscan=100.0)

            insert_validation_result(file.filename, "Valid" if result["status"] == "success" else "Invalid", result.get("reason", ""), username, client_ip)

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            insert_validation_result(file.filename, "Invalid", str(e), username, client_ip)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Deleted temporary file: {temp_file_path}")

    return JSONResponse(content={"message": "Image validation completed."})

@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Image Validation API"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
