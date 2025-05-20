from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import numpy as np
import random
import requests
import time
import cv2
import os
import re
import sys
import boto3
from boto3.dynamodb.conditions import Key
from google.cloud import firestore
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from functools import lru_cache
import os
import uuid
from datetime import datetime
import subprocess
from tempfile import NamedTemporaryFile
import json

# Add new Pydantic models
class StorageResult(BaseModel):
    cid: str
    filename: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

app = FastAPI(title="Face Recognition API", description="API for face recognition with multiple database options")

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
CASCADE = "face_cascade.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE)

# Add these models after the existing RecognitionResult model
class ConsensusCheck(BaseModel):
    person_name: str
    check_timestamp: float = Field(default_factory=time.time)

class ConsensusCheckResult(BaseModel):
    found: bool
    timestamp: Optional[float] = None

# Hardcoded consensus configuration
CONSENSUS_HOSTS = ["http://localhost:8001", "http://localhost:8002"]
CONSENSUS_THRESHOLD = 0.6  # 60% consensus required
MOCK_ENABLED = True  # Enable mock responses
MOCK_TRUE_PROBABILITY = 0.8  # 80% chance of true responses when mocking

# Configuration
class Settings(BaseModel):
    db_provider: str = os.getenv("DB_PROVIDER", "local")  # "local", "dynamodb", or "firestore"
    dynamo_table_name: str = os.getenv("DYNAMO_TABLE_NAME", "FaceRecognitionData")
    dynamo_results_table_name: str = os.getenv("DYNAMO_RESULTS_TABLE_NAME", "FaceRecognitionResults")
    dynamo_region: str = os.getenv("AWS_REGION", "us-east-1")
    local_db_path: str = os.getenv("LOCAL_DB_PATH", "Data")
    local_results_path: str = os.getenv("LOCAL_RESULTS_PATH", "RecognitionResults")
    storacha_script_path: str = os.getenv("STORACHA_SCRIPT_PATH", "./storacha.js")
    storacha_key: str = os.getenv("STORACHA_KEY", "MgCYaFfwSPxJuwYi4t3+rhNnyJs+e1GCSv2qDbaYnXg0KHO0BdOcDgI8+K4ByU5m64nEyOpNkF1A7tLfSc1MiRQftrIc=")
    # In Settings class
 #   storacha_key: str = os.getenv("STORACHA_KEY", "CC-PART2:did:key:z6MkgMomqWhzTsUdf59yYXXehyQxcYmztD5JmhLyUNvSoDbe")

    
@lru_cache()
def get_settings():
    return Settings()

# Database interface
class DatabaseProvider:
    def get_data(self) -> List[str]:
        """Get face recognition training data"""
        pass
    
    def save_data(self, data_list: List[str]) -> None:
        """Save face recognition training data"""
        pass
    
    def save_recognition_result(self, result: Dict[str, Any]) -> str:
        """Save a recognition result and return its ID"""
        pass
    
    def get_recognition_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent recognition results"""
        pass

# Local (shelve) implementation
class LocalDatabase(DatabaseProvider):
    def __init__(self, settings: Settings):
        self.db_path = settings.local_db_path
        self.results_path = settings.local_results_path
    
    def get_data(self) -> List[str]:
        import shelve
        with shelve.open(self.db_path) as datafile:
            return datafile.get("Data", [])
    
    def save_data(self, data_list: List[str]) -> None:
        import shelve
        with shelve.open(self.db_path) as datafile:
            datafile["Data"] = data_list
    
    def save_recognition_result(self, result: Dict[str, Any]) -> str:
        import shelve
        
        # Add timestamp and unique ID
        result_id = str(uuid.uuid4())
        result["id"] = result_id
        result["timestamp"] = datetime.now().isoformat()
        
        with shelve.open(self.results_path) as results_file:
            # Get existing results
            existing_results = results_file.get("Results", [])
            
            # Add new result
            existing_results.append(result)
            
            # Only keep the last 1000 results
            if len(existing_results) > 1000:
                existing_results = existing_results[-1000:]
                
            # Save back to the shelve
            results_file["Results"] = existing_results
        
        return result_id
    
    def get_recognition_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        import shelve
        try:
            with shelve.open(self.results_path) as results_file:
                results = results_file.get("Results", [])
                # Return most recent first, limited
                return sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
        except Exception as e:
            print(f"Error getting recognition results: {e}")
            return []

# Add Node.js helper functions
async def run_node_script(script_path: str, args: list):
    try:
        result = subprocess.run(
            ["node", script_path] + args,
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Storacha error: {e.stderr}"
        )

# DynamoDB implementation
class DynamoDBDatabase(DatabaseProvider):
    def __init__(self, settings: Settings):
        self.table_name = settings.dynamo_table_name
        self.results_table_name = settings.dynamo_results_table_name
        self.region = settings.dynamo_region
        
        # Configure AWS SDK
        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.client = boto3.client('dynamodb', region_name=self.region)
        
        # Ensure tables exist - with better error handling
        self._create_table_if_not_exists(self.table_name, 'id')
        self._create_table_if_not_exists(self.results_table_name, 'id', sort_key='timestamp')
        
        # Get the table references
        self.table = self.dynamodb.Table(self.table_name)
        self.results_table = self.dynamodb.Table(self.results_table_name)
        
    def _create_table_if_not_exists(self, table_name, partition_key, sort_key=None):
        """Create the DynamoDB table if it doesn't exist"""
        try:
            # Check if table exists
            existing_tables = self.client.list_tables()['TableNames']
            
            if table_name not in existing_tables:
                print(f"Creating DynamoDB table {table_name}...")
                
                # Set up key schema
                key_schema = [{'AttributeName': partition_key, 'KeyType': 'HASH'}]
                attribute_definitions = [{'AttributeName': partition_key, 'AttributeType': 'S'}]
                
                # Add sort key if provided
                if sort_key:
                    key_schema.append({'AttributeName': sort_key, 'KeyType': 'RANGE'})
                    attribute_definitions.append({'AttributeName': sort_key, 'AttributeType': 'S'})
                
                # Create the table with proper parameters
                self.client.create_table(
                    TableName=table_name,
                    KeySchema=key_schema,
                    AttributeDefinitions=attribute_definitions,
                    ProvisionedThroughput={
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                )
                
                # Wait for the table to be created before proceeding
                print(f"Waiting for DynamoDB table {table_name} to become active...")
                waiter = self.client.get_waiter('table_exists')
                waiter.wait(TableName=table_name)
                print(f"Table {table_name} is now active.")
                
        except Exception as e:
            print(f"Error creating DynamoDB table {table_name}: {e}")
            # Check for specific AWS errors
            if hasattr(e, 'response') and 'Error' in getattr(e, 'response', {}):
                error_code = e.response['Error'].get('Code', '')
                error_message = e.response['Error'].get('Message', '')
                print(f"AWS Error Code: {error_code}, Message: {error_message}")
    
    def get_data(self) -> List[str]:
        try:
            response = self.table.get_item(Key={'id': 'face_data'})
            if 'Item' in response:
                return response['Item'].get('data', [])
            return []
        except Exception as e:
            print(f"Error getting data from DynamoDB: {e}")
            return []
    
    def save_data(self, data_list: List[str]) -> None:
        try:
            self.table.put_item(Item={'id': 'face_data', 'data': data_list})
        except Exception as e:
            print(f"Error saving data to DynamoDB: {e}")
    
    def save_recognition_result(self, result: Dict[str, Any]) -> str:
        try:
            # Add timestamp and unique ID
            result_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Prepare item for DynamoDB
            item = {
                'id': result_id,
                'timestamp': timestamp,
                'names': result.get('recognized_names', []),
                'db_provider': result.get('db_provider', ''),
                'confidence_scores': result.get('confidence_scores', {}),
            }
            
            # Add any additional data from the result
            for key, value in result.items():
                if key not in item:
                    item[key] = value
            
            # Store in DynamoDB
            self.results_table.put_item(Item=item)
            
            return result_id
        except Exception as e:
            print(f"Error saving recognition result to DynamoDB: {e}")
            return None
    
    def get_recognition_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            # Note: This is a simple implementation that doesn't use GSIs
            # For production, you might want to create a GSI on timestamp for efficient queries
            response = self.results_table.scan(Limit=limit)
            items = response.get('Items', [])
            
            # Sort by timestamp (newest first)
            items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return items[:limit]
        except Exception as e:
            print(f"Error getting recognition results from DynamoDB: {e}")
            return []

# Google Firestore implementation
class FirestoreDatabase(DatabaseProvider):
    def __init__(self, settings: Settings):
        self.db = firestore.Client()
        self.collection = 'face_recognition'
        self.doc_id = 'face_data'
        self.results_collection = 'recognition_results'
    
    def get_data(self) -> List[str]:
        try:
            doc_ref = self.db.collection(self.collection).document(self.doc_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict().get('data', [])
            return []
        except Exception as e:
            print(f"Error getting data from Firestore: {e}")
            return []
    
    def save_data(self, data_list: List[str]) -> None:
        try:
            doc_ref = self.db.collection(self.collection).document(self.doc_id)
            doc_ref.set({'data': data_list})
        except Exception as e:
            print(f"Error saving data to Firestore: {e}")
    
    def save_recognition_result(self, result: Dict[str, Any]) -> str:
        try:
            # Add timestamp and generate ID
            result_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create a new document with auto-generated ID
            result_doc = {
                'id': result_id,
                'timestamp': timestamp,
                'recognized_names': result.get('recognized_names', []),
                'db_provider': result.get('db_provider', ''),
            }
            
            # Add any additional data from the result
            for key, value in result.items():
                if key not in result_doc:
                    result_doc[key] = value
            
            # Save to Firestore
            self.db.collection(self.results_collection).document(result_id).set(result_doc)
            
            return result_id
        except Exception as e:
            print(f"Error saving recognition result to Firestore: {e}")
            return None
    
    def get_recognition_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            # Query the results collection, ordered by timestamp descending, limited
            query = (self.db.collection(self.results_collection)
                     .order_by('timestamp', direction=firestore.Query.DESCENDING)
                     .limit(limit))
            
            results = []
            for doc in query.stream():
                data = doc.to_dict()
                # Convert timestamp to string for consistency across providers
                if isinstance(data.get('timestamp'), datetime):
                    data['timestamp'] = data['timestamp'].isoformat()
                results.append(data)
                
            return results
        except Exception as e:
            print(f"Error getting recognition results from Firestore: {e}")
            return []

# Factory to get the appropriate database provider
def get_db_provider(settings: Settings = Depends(get_settings)) -> DatabaseProvider:
    if settings.db_provider == "dynamodb":
        return DynamoDBDatabase(settings)
    elif settings.db_provider == "firestore":
        return FirestoreDatabase(settings)
    else:
        return LocalDatabase(settings)

# Face recognition functions
def get_images(path, settings=None):
    images = []
    labels = []
    count = 0
    
    # Check if the dataset directory exists
    if not os.path.exists(path):
        print(f"Dataset directory {path} does not exist. Creating it...")
        os.makedirs(path)
    
    if len(os.listdir(path)) == 0:
        print("Empty Dataset.......aborting Training")
        sys.exit()
    
    # Get data list from the database
    if settings is None:
        settings = get_settings()
        
    # Create the appropriate database provider directly
    if settings.db_provider == "dynamodb":
        db_provider = DynamoDBDatabase(settings)
    elif settings.db_provider == "firestore":
        db_provider = FirestoreDatabase(settings)
    else:
        db_provider = LocalDatabase(settings)
        
    data_list = db_provider.get_data()
    
    for img in os.listdir(path):
        regex = re.compile(r'(\d+|\s+)')
        labl = regex.split(img)[0]
        if labl not in data_list:
            data_list.append(labl)
        count += 1
        image_path = os.path.join(path, img)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image_grey)
        labels.append(data_list.index(labl))
    
    # Save updated data list back to database
    db_provider.save_data(data_list)
    
    return images, labels, count

def initialize_recognizer():
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        try:
            face_recognizer = cv2.createLBPHFaceRecognizer()
        except Exception as e:
            print(f"Error creating face recognizer: {e}")
            return None
            
    print("Training..........")
    settings = get_settings()
    
    # Create Dataset directory if it doesn't exist
    if not os.path.exists("./Dataset"):
        print("Dataset directory does not exist. Creating it...")
        os.makedirs("./Dataset")
        print("Please add images to the Dataset directory and then call /train endpoint")
        return face_recognizer
        
    if len(os.listdir("./Dataset")) == 0:
        print("Empty Dataset. Skipping training for now.")
        return face_recognizer
        
    try:
        dataset = get_images("./Dataset", settings)
        print(f"Recognizer trained using Dataset: {dataset[2]} Images used")
        face_recognizer.train(dataset[0], np.array(dataset[1]))
    except Exception as e:
        print(f"Error during training: {e}")
    
    return face_recognizer

# Initialize face recognizer at startup
face_recognizer = None  # Will be initialized properly later

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    """Upload a file to Storacha"""
    try:
        # Save uploaded file to temp location
        with NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Call Node.js upload script
        result = await run_node_script(
            settings.storacha_script_path,
            ["upload", tmp_path, settings.storacha_key]
        )

        # Store CID in database (modify your database classes to handle this)
        db_provider = get_db_provider(settings)
        db_provider.save_storage_reference(result['cid'], file.filename)

        os.unlink(tmp_path)
        return result

    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{cid}")
async def download_file(
    cid: str,
    settings: Settings = Depends(get_settings)
):
    """Download a file from Storacha by CID"""
    try:
        result = await run_node_script(
            settings.storacha_script_path,
            ["download", cid, settings.storacha_key]
        )
        return RedirectResponse(url=result['gateway_url'])
    except Exception as e:
        raise HTTPException(status_code=400)

# Pydantic model for recognition result
class RecognitionResult(BaseModel):
    recognized_names: List[str]
    db_provider: str
    confidence_scores: Dict[str, float] = {}
    timestamp: Optional[str] = None
    id: Optional[str] = None

@app.post("/recognize", response_model=RecognitionResult)
async def recognize_image(
    file: UploadFile = File(...),
    db_provider: DatabaseProvider = Depends(get_db_provider)
):
    global face_recognizer
    
    if face_recognizer is None:
        raise HTTPException(status_code=500, detail="Face recognizer not initialized. Please call /train endpoint first.")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File is not an image.")
    
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    
    # Get data list from the database
    data_list = db_provider.get_data()
    
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0
    )
    
    if len(faces) == 0:
        result = {
            "recognized_names": [],
            "message": "No faces detected.",
            "db_provider": get_settings().db_provider,
            "confidence_scores": {}
        }
        # Save the empty result
        result_id = db_provider.save_recognition_result(result)
        result["id"] = result_id
        return JSONResponse(content=result)
    
    detected_names = set()
    confidence_scores = {}
    
    for (x, y, w, h) in faces:
        sub_img = image_grey[y:y + h, x:x + w]
        try:
            nbr, conf = face_recognizer.predict(sub_img)
            if nbr < len(data_list):
                name = data_list[nbr]
                trimmed_name = name.replace("_", "").replace(" ", "").strip()
                detected_names.add(trimmed_name)
                confidence_scores[trimmed_name] = float(conf)
                print(f"Recognized {trimmed_name} with confidence {conf}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue
    
    # Create the result object
    recognized_names = list(detected_names)
    result = {
        "recognized_names": recognized_names,
        "db_provider": get_settings().db_provider,
        "confidence_scores": confidence_scores
    }
    
    # If we have recognized faces and consensus hosts are configured, check consensus
    if recognized_names and CONSENSUS_HOSTS:
        print(f"Checking consensus among {len(CONSENSUS_HOSTS)} hosts")
        consensus_results = {}
        
        # For each recognized person, check if other hosts also detected them
        for person_name in recognized_names:
            consensus_results[person_name] = 1  # Count self as having detected
            
            # Query each other host
            for host_url in CONSENSUS_HOSTS:
                try:
                    check_data = ConsensusCheck(
                        person_name=person_name
                    )
                    
                    response = requests.post(
                        f"{host_url}/consensus-check",
                        json=check_data.dict(),
                        timeout=2  # Short timeout to prevent blocking
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        if response_data.get("found", False):
                            consensus_results[person_name] += 1
                            
                except Exception as e:
                    print(f"Error checking consensus with host {host_url}: {e}")
        
        # Calculate consensus percentages and filter recognized names
        total_hosts = len(CONSENSUS_HOSTS) + 1  # +1 for self
        consensus_threshold = total_hosts * CONSENSUS_THRESHOLD
        
        # Filter to keep only names that meet the consensus threshold
        consensus_names = [
            name for name in recognized_names 
            if consensus_results.get(name, 0) >= consensus_threshold
        ]
        
        # Update the recognized names to only those with consensus
        result["recognized_names"] = consensus_names
        result["consensus_info"] = {
            "total_hosts": total_hosts,
            "threshold": CONSENSUS_THRESHOLD,
            "results": {name: count/total_hosts for name, count in consensus_results.items()}
        }
    
    # Only save the recognition result if there are recognized faces after consensus
    if result["recognized_names"]:
        result_id = db_provider.save_recognition_result(result)
        result["id"] = result_id
    
    return JSONResponse(content=result)

@app.post("/consensus-check")
async def consensus_check(
    check_data: ConsensusCheck,
    db_provider: DatabaseProvider = Depends(get_db_provider)
):
    """Check if a person was recognized in the last 5 seconds"""
    # If mock is enabled, return random result based on probability
    if MOCK_ENABLED:
        random_result = random.random() < MOCK_TRUE_PROBABILITY
        return ConsensusCheckResult(found=random_result, timestamp=time.time())
    
    # Get recent recognition results
    results = db_provider.get_recognition_results(limit=50)
    
    # Filter results from the last 5 seconds
    current_time = time.time()
    five_seconds_ago = current_time - 5
    
    for result in results:
        # Skip results without timestamp
        if "timestamp" not in result:
            continue
            
        # Convert ISO timestamp string to time
        try:
            result_time = datetime.fromisoformat(result["timestamp"]).timestamp()
        except (ValueError, TypeError):
            continue
            
        # Check if result is within the last 5 seconds
        if result_time >= five_seconds_ago:
            # Check if the person was recognized
            recognized_names = result.get("recognized_names", [])
            if check_data.person_name in recognized_names:
                return ConsensusCheckResult(found=True, timestamp=result_time)
    
    return ConsensusCheckResult(found=False)

@app.get("/recognition-results", response_model=List[RecognitionResult])
async def get_recognition_results(
    limit: int = 100,
    db_provider: DatabaseProvider = Depends(get_db_provider)
):
    """Get recent recognition results"""
    results = db_provider.get_recognition_results(limit=limit)
    return JSONResponse(content=results)

@app.post("/train")
async def train_model(settings: Settings = Depends(get_settings)):
    """Retrain the face recognition model"""
    global face_recognizer
    try:
        face_recognizer = initialize_recognizer()
        if face_recognizer is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to initialize face recognizer"}
            )
        
        return JSONResponse(content={"message": "Model trained successfully", "db_provider": settings.db_provider})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error training model: {str(e)}"}
        )

@app.get("/db-info/")
async def get_db_info(settings: Settings = Depends(get_settings)):
    """Get information about the current database configuration and status"""
    info = {
        "current_db_provider": settings.db_provider
    }
    
    if settings.db_provider == "dynamodb":
        try:
            # Check DynamoDB table status
            dynamodb = boto3.client('dynamodb', region_name=settings.dynamo_region)
            table_info = dynamodb.describe_table(TableName=settings.dynamo_table_name)
            results_table_info = dynamodb.describe_table(TableName=settings.dynamo_results_table_name)
            
            info["training_table"] = {
                "name": settings.dynamo_table_name,
                "status": table_info["Table"]["TableStatus"]
            }
            
            info["results_table"] = {
                "name": settings.dynamo_results_table_name,
                "status": results_table_info["Table"]["TableStatus"]
            }
            
            info["region"] = settings.dynamo_region
        except Exception as e:
            info["error"] = str(e)
            # Check for specific AWS errors
            if hasattr(e, 'response') and 'Error' in getattr(e, 'response', {}):
                error_code = e.response['Error'].get('Code', '')
                error_message = e.response['Error'].get('Message', '')
                info["error_code"] = error_code
                info["error_message"] = error_message
    
    elif settings.db_provider == "firestore":
        info["training_collection"] = {
            "collection": "face_recognition",
            "document_id": "face_data"
        }
        info["results_collection"] = {
            "collection": "recognition_results"
        }
        
    else:  # local
        info["training_db_path"] = settings.local_db_path
        info["results_db_path"] = settings.local_results_path
        
    return info

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Create a separate function to initialize the app, avoiding startup errors
def initialize_app():
    global face_recognizer
    
    # Ensure the Dataset directory exists
    if not os.path.exists("./Dataset"):
        print("Dataset directory does not exist. Creating it...")
        os.makedirs("./Dataset")
    
    try:
        face_recognizer = initialize_recognizer()
        if face_recognizer is None:
            print("Warning: Face recognizer not initialized properly. Please call /train endpoint.")
    except Exception as e:
        print(f"Warning: Error initializing recognizer: {e}")
        print("The application will start, but you may need to call /train endpoint manually")

# Initialize at startup
initialize_app()

# To run the server, save this as `api_server.py` and run:
# uvicorn api_server:app --reload

# To switch database providers, set the DB_PROVIDER environment variable:
# DB_PROVIDER=local (default), DB_PROVIDER=dynamodb, or DB_PROVIDER=firestore