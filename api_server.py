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
import ipfshttpclient
import base64
import io
import tempfile

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
MOCK_TRUE_PROBABILITY = 0.4  # 80% chance of true responses when mocking

# Configuration
class Settings(BaseModel):
    db_provider: str = os.getenv("DB_PROVIDER", "local")  # "local", "dynamodb", or "firestore"
    dynamo_table_name: str = os.getenv("DYNAMO_TABLE_NAME", "FaceRecognitionData")
    dynamo_results_table_name: str = os.getenv("DYNAMO_RESULTS_TABLE_NAME", "FaceRecognitionResults")
    dynamo_region: str = os.getenv("AWS_REGION", "us-east-1")
    local_db_path: str = os.getenv("LOCAL_DB_PATH", "Data")
    local_results_path: str = os.getenv("LOCAL_RESULTS_PATH", "RecognitionResults")
    ipfs_host: str = os.getenv("IPFS_HOST", "/ip4/127.0.0.1/tcp/5001")
    ipfs_data_hash_file: str = os.getenv("IPFS_DATA_HASH_FILE", "ipfs_data_hash.txt")
    ipfs_results_index_file: str = os.getenv("IPFS_RESULTS_INDEX_FILE", "ipfs_results_index.txt")
    
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

class IPFSDatabase(DatabaseProvider):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ipfs_host = settings.ipfs_host
        self.data_hash_file = settings.ipfs_data_hash_file
        self.results_index_file = settings.ipfs_results_index_file
        
        # Create or load the results index file if it doesn't exist
        if not os.path.exists(self.results_index_file):
            with open(self.results_index_file, 'w') as f:
                json.dump([], f)
    
    def _get_ipfs_client(self):
        """Connect to IPFS daemon"""
        try:
            return ipfshttpclient.connect(self.ipfs_host)
        except Exception as e:
            print(f"Error connecting to IPFS daemon: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to IPFS daemon: {str(e)}")
    
    def _load_data_hash(self):
        """Load the IPFS hash for the data file"""
        if not os.path.exists(self.data_hash_file):
            return None
        
        try:
            with open(self.data_hash_file, 'r') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading data hash file: {e}")
            return None
    
    def _save_data_hash(self, ipfs_hash):
        """Save the IPFS hash for the data file"""
        try:
            with open(self.data_hash_file, 'w') as f:
                f.write(ipfs_hash)
        except Exception as e:
            print(f"Error saving data hash: {e}")
    
    def _load_results_index(self):
        """Load the results index from local file"""
        try:
            with open(self.results_index_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results index: {e}")
            return []
    
    def _save_results_index(self, index):
        """Save the results index to local file"""
        try:
            with open(self.results_index_file, 'w') as f:
                json.dump(index, f)
        except Exception as e:
            print(f"Error saving results index: {e}")
    
    def get_data(self) -> List[str]:
        """Get face recognition training data from IPFS"""
        data_hash = self._load_data_hash()
        
        if not data_hash:
            return []
        
        try:
            # Use HTTP API directly
            api_url = f"http://127.0.0.1:5001/api/v0/cat?arg={data_hash}"
            response = requests.post(api_url)
            if response.status_code != 200:
                raise Exception(f"Failed to cat IPFS hash: {response.status_code}")
            
            # Parse the JSON data
            structured_data = json.loads(response.content)
            
            # Check if it's already in the structured format
            if isinstance(structured_data, dict) and "names" in structured_data:
                return structured_data["names"]
            
            # If it's still the old format (just a list), return it as is
            return structured_data
        except Exception as e:
            print(f"Error getting data from IPFS: {e}")
            return []
    
    def save_data(self, data_list: List[str]) -> None:
        """Save face recognition training data to IPFS with timestamps"""
        try:
            # Create a structured data object with timestamps
            structured_data = {
                "names": data_list,
                "updated_at": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(),
                "record_count": len(data_list)
            }
            
            # Convert data to JSON
            json_data = json.dumps(structured_data)
            
            # Use HTTP API directly
            api_url = "http://127.0.0.1:5001/api/v0/add"
            files = {'file': ('data.json', json_data)}
            response = requests.post(api_url, files=files)
            
            if response.status_code != 200:
                raise Exception(f"Failed to add to IPFS: {response.status_code}")
            
            # Parse the response to get the hash
            result = response.json()
            res = result.get('Hash')
            
            # Save the hash to the local file
            self._save_data_hash(res)
            
            print(f"Data saved to IPFS with hash: {res}")
        except Exception as e:
            print(f"Error saving data to IPFS: {e}")
    
    def save_recognition_result(self, result: Dict[str, Any]) -> str:
        """Save a recognition result to IPFS and return its ID"""
        try:
            # Add timestamp and unique ID
            result_id = str(uuid.uuid4())
            result["id"] = result_id
            result["timestamp"] = datetime.now().isoformat()
            
            # Convert to JSON
            json_data = json.dumps(result)
            
            # Use HTTP API directly
            api_url = "http://127.0.0.1:5001/api/v0/add"
            files = {'file': ('result.json', json_data)}
            response = requests.post(api_url, files=files)
            
            if response.status_code != 200:
                raise Exception(f"Failed to add to IPFS: {response.status_code}")
            
            # Get the IPFS hash
            ipfs_hash = response.json().get('Hash')
            
            # Update results index
            results_index = self._load_results_index()
            results_index.append({
                "id": result_id,
                "ipfs_hash": ipfs_hash,
                "timestamp": result["timestamp"]
            })
            self._save_results_index(results_index)
            
            return result_id
        except Exception as e:
            print(f"Error saving recognition result to IPFS: {e}")
            return None
    
    def get_recognition_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent recognition results from IPFS"""
        try:
            # Connect to IPFS
            client = self._get_ipfs_client()
            
            # Load the results index
            results_index = self._load_results_index()
            
            # Sort by timestamp (newest first)
            sorted_index = sorted(results_index, key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Limit the number of results
            limited_index = sorted_index[:limit]
            
            results = []
            for item in limited_index:
                try:
                    # Get the result from IPFS
                    ipfs_hash = item["ipfs_hash"]
                    result_data = client.get_json(ipfs_hash)
                    results.append(result_data)
                except Exception as e:
                    print(f"Error getting result from IPFS (hash: {item.get('ipfs_hash')}): {e}")
                    continue
            
            return results
        except Exception as e:
            print(f"Error getting recognition results from IPFS: {e}")
            return []

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
    elif settings.db_provider == "ipfs":
        return IPFSDatabase(settings)
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
    elif settings.db_provider == "ipfs":
        db_provider = IPFSDatabase(settings)
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
        return None  # Return None instead of untrained model
        
    if len(os.listdir("./Dataset")) == 0:
        print("Empty Dataset. Skipping training for now.")
        return None  # Return None instead of untrained model
        
    try:
        dataset = get_images("./Dataset", settings)
        print(f"Recognizer trained using Dataset: {dataset[2]} Images used")
        face_recognizer.train(dataset[0], np.array(dataset[1]))
        return face_recognizer
    except Exception as e:
        print(f"Error during training: {e}")
        return None  # Return None to indicate failure

# Initialize face recognizer at startup
face_recognizer = None  # Will be initialized properly later

# Pydantic model for recognition result
class RecognitionResult(BaseModel):
    recognized_names: List[str]
    db_provider: str
    confidence_scores: Dict[str, float] = {}
    timestamp: Optional[str] = None
    id: Optional[str] = None

@app.get("/ipfs-info")
async def get_ipfs_info(settings: Settings = Depends(get_settings)):
    """Get information about IPFS configuration"""
    if settings.db_provider != "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "IPFS is not configured as the active database provider"}
        )
    
    try:
        # Connect to IPFS
        client = ipfshttpclient.connect(settings.ipfs_host)
        
        # Get IPFS ID information
        id_info = client.id()
        
        # Get data hash if available
        ipfs_db = IPFSDatabase(settings)
        data_hash = ipfs_db._load_data_hash()
        
        # Get results index
        results_index = ipfs_db._load_results_index()
        
        return {
            "ipfs_host": settings.ipfs_host,
            "data_hash_file": settings.ipfs_data_hash_file,
            "results_index_file": settings.ipfs_results_index_file,
            "ipfs_version": id_info.get("AgentVersion", "Unknown"),
            "ipfs_id": id_info.get("ID", "Unknown"),
            "data_hash": data_hash,
            "results_count": len(results_index)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error connecting to IPFS: {str(e)}"}
        )

@app.get("/ipfs-pin/{ipfs_hash}")
async def pin_ipfs_hash(ipfs_hash: str, settings: Settings = Depends(get_settings)):
    """Pin an IPFS hash to ensure it remains available"""
    if settings.db_provider != "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "IPFS is not configured as the active database provider"}
        )
    
    try:
        # Connect to IPFS
        client = ipfshttpclient.connect(settings.ipfs_host)
        
        # Pin the hash
        pin_info = client.pin.add(ipfs_hash)
        
        return {
            "status": "success",
            "message": f"Successfully pinned {ipfs_hash}",
            "pins": pin_info["Pins"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error pinning IPFS hash: {str(e)}"}
        )

@app.post("/upload-to-ipfs")
async def upload_to_ipfs(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    """Upload a file directly to IPFS and return its hash"""
    if settings.db_provider != "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "IPFS is not configured as the active database provider"}
        )
    
    try:
        # Read the file
        contents = await file.read()
        
        # Connect to IPFS
        client = ipfshttpclient.connect(settings.ipfs_host)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            temp.write(contents)
        
        try:
            # Add the file to IPFS
            res = client.add(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return {
                "status": "success",
                "filename": file.filename,
                "ipfs_hash": res["Hash"],
                "size": res["Size"]
            }
        except Exception as e:
            # Clean up temporary file in case of exception
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error uploading to IPFS: {str(e)}"}
        )

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
    elif settings.db_provider == "ipfs":
        try:
            ipfs_db = IPFSDatabase(settings)
            data_hash = ipfs_db._load_data_hash()
            results_index = ipfs_db._load_results_index()
            
            info["ipfs_host"] = settings.ipfs_host
            info["data_hash_file"] = settings.ipfs_data_hash_file
            info["results_index_file"] = settings.ipfs_results_index_file
            info["data_hash"] = data_hash
            info["results_count"] = len(results_index)
            
            # Try to connect to IPFS to verify it's working
            try:
                client = ipfshttpclient.connect(settings.ipfs_host)
                id_info = client.id()
                info["connection_status"] = "connected"
                info["ipfs_version"] = id_info.get("AgentVersion", "Unknown")
            except Exception as e:
                info["connection_status"] = "error"
                info["connection_error"] = str(e)
        
        except Exception as e:
            info["error"] = str(e)
        
    else:  # local
        info["training_db_path"] = settings.local_db_path
        info["results_db_path"] = settings.local_results_path
        
    return info

@app.post("/export-to-ipfs")
async def export_to_ipfs(
    db_provider: DatabaseProvider = Depends(get_db_provider),
    settings: Settings = Depends(get_settings)
):
    """Export all data and recognition results to IPFS"""
    if settings.db_provider == "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "Current database provider is already IPFS"}
        )
    
    try:
        # Get all data from the current database
        data_list = db_provider.get_data()
        recognition_results = db_provider.get_recognition_results(limit=1000)
        
        # Connect to IPFS
        client = ipfshttpclient.connect(settings.ipfs_host)
        
        # Export data to IPFS
        data_hash = client.add_json(data_list)
        
        # Export each recognition result to IPFS and build an index
        results_index = []
        for result in recognition_results:
            result_id = result.get("id") or str(uuid.uuid4())
            result_hash = client.add_json(result)
            
            results_index.append({
                "id": result_id,
                "ipfs_hash": result_hash,
                "timestamp": result.get("timestamp", datetime.now().isoformat())
            })
        
        # Create a temporary IPFS database to save the index
        ipfs_db = IPFSDatabase(settings)
        ipfs_db._save_data_hash(data_hash)
        ipfs_db._save_results_index(results_index)
        
        return {
            "status": "success",
            "data_hash": data_hash,
            "results_count": len(results_index),
            "note": "To use this data, set DB_PROVIDER=ipfs in your environment"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error exporting to IPFS: {str(e)}"}
        )

@app.post("/import-from-ipfs")
async def import_from_ipfs(
    db_provider: DatabaseProvider = Depends(get_db_provider),
    settings: Settings = Depends(get_settings)
):
    """Import data and recognition results from IPFS to current database provider"""
    if settings.db_provider == "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "Current database provider is already IPFS"}
        )
    
    try:
        # Create a temporary IPFS database to get data
        ipfs_db = IPFSDatabase(Settings(db_provider="ipfs"))
        
        # Get data and results from IPFS
        data_list = ipfs_db.get_data()
        recognition_results = ipfs_db.get_recognition_results(limit=1000)
        
        # Import data to current database
        db_provider.save_data(data_list)
        
        # Import recognition results to current database
        imported_count = 0
        for result in recognition_results:
            db_provider.save_recognition_result(result)
            imported_count += 1
        
        return {
            "status": "success",
            "imported_data_count": len(data_list),
            "imported_results_count": imported_count
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error importing from IPFS: {str(e)}"}
        )

@app.get("/view-data-in-ipfs")
async def view_data_in_ipfs(settings: Settings = Depends(get_settings)):
    if settings.db_provider != "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "IPFS is not configured as the active database provider"}
        )
    
    ipfs_db = IPFSDatabase(settings)
    data_hash = ipfs_db._load_data_hash()
    
    if not data_hash:
        return JSONResponse(
            status_code=404,
            content={"error": "No data hash found"}
        )
    
    # Create gateway URL for viewing the data
    gateway_url = f"http://127.0.0.1:8080/ipfs/{data_hash}"
    
    return {
        "data_hash": data_hash,
        "view_url": gateway_url,
        "webui_url": f"http://127.0.0.1:5001/explore/ipfs/{data_hash}"
    }

@app.get("/view-recognition-results-in-ipfs")
async def view_recognition_results_in_ipfs(settings: Settings = Depends(get_settings)):
    if settings.db_provider != "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "IPFS is not configured as the active database provider"}
        )
    
    ipfs_db = IPFSDatabase(settings)
    results_index = ipfs_db._load_results_index()
    
    if not results_index:
        return JSONResponse(
            status_code=404,
            content={"error": "No recognition results found"}
        )
    
    # Create a list of results with their view URLs
    result_urls = []
    for result in results_index:
        ipfs_hash = result.get("ipfs_hash")
        if ipfs_hash:
            result_urls.append({
                "id": result.get("id", "unknown"),
                "timestamp": result.get("timestamp", "unknown"),
                "ipfs_hash": ipfs_hash,
                "view_url": f"http://127.0.0.1:8080/ipfs/{ipfs_hash}",
                "webui_url": f"http://127.0.0.1:5001/webui/#/explore/ipfs/{ipfs_hash}"
            })
    
    return {
        "result_count": len(result_urls),
        "results": result_urls
    }

@app.post("/train")
async def train_model(settings: Settings = Depends(get_settings)):
    """Retrain the face recognition model"""
    global face_recognizer
    try:
        # Check if Dataset directory has images
        if not os.path.exists("./Dataset") or len(os.listdir("./Dataset")) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Dataset directory is empty or doesn't exist. Please add face images first."
                }
            )
            
        face_recognizer = initialize_recognizer()
        if face_recognizer is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to train model. Check server logs for details."}
            )
        
        return JSONResponse(content={"message": "Model trained successfully", "db_provider": settings.db_provider})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error training model: {str(e)}"}
        )
        
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/all-recognition-results")
async def get_all_recognition_results(settings: Settings = Depends(get_settings)):
    """Get all recognition results ever saved to IPFS"""
    if settings.db_provider != "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "This endpoint is only available when using IPFS as the database provider"}
        )
    
    try:
        # Create IPFS database
        ipfs_db = IPFSDatabase(settings)
        
        # Load all results without limit
        results = ipfs_db.get_recognition_results(limit=10000)  # Increase this number as needed
        
        # Group results by date for easier navigation
        grouped_results = {}
        for result in results:
            timestamp = result.get("timestamp", "unknown")
            date = timestamp.split("T")[0] if "T" in timestamp else "unknown"
            
            if date not in grouped_results:
                grouped_results[date] = []
            
            grouped_results[date].append(result)
        
        # Count total recognitions by person
        person_counts = {}
        for result in results:
            for person in result.get("recognized_names", []):
                if person not in person_counts:
                    person_counts[person] = 0
                person_counts[person] += 1
        
        return {
            "total_results": len(results),
            "person_counts": person_counts,
            "results_by_date": grouped_results
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting all results: {str(e)}"}
        )

@app.get("/export-results")
async def export_results(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    person: Optional[str] = None,
    settings: Settings = Depends(get_settings)
):
    """Export recognition results within a date range"""
    if settings.db_provider != "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "This endpoint is only available when using IPFS as the database provider"}
        )
    
    try:
        # Create IPFS database
        ipfs_db = IPFSDatabase(settings)
        
        # Load all results
        all_results = ipfs_db.get_recognition_results(limit=10000)
        
        # Filter by date range if provided
        filtered_results = all_results
        if start_date:
            filtered_results = [
                r for r in filtered_results 
                if r.get("timestamp", "").startswith(start_date) or r.get("timestamp", "") > start_date
            ]
        
        if end_date:
            filtered_results = [
                r for r in filtered_results 
                if r.get("timestamp", "").startswith(end_date) or r.get("timestamp", "") < end_date
            ]
            
        # Filter by person if provided
        if person:
            filtered_results = [
                r for r in filtered_results 
                if person in r.get("recognized_names", [])
            ]
        
        # Save results to a new IPFS file
        api_url = "http://127.0.0.1:5001/api/v0/add"
        export_data = {
            "results": filtered_results,
            "query": {
                "start_date": start_date,
                "end_date": end_date,
                "person": person
            },
            "count": len(filtered_results),
            "export_time": datetime.now().isoformat()
        }
        
        json_data = json.dumps(export_data)
        files = {'file': ('export.json', json_data)}
        response = requests.post(api_url, files=files)
        
        if response.status_code != 200:
            raise Exception(f"Failed to add to IPFS: {response.status_code}")
        
        # Parse the response to get the hash
        result = response.json()
        export_hash = result.get('Hash')
        
        return {
            "status": "success",
            "result_count": len(filtered_results),
            "ipfs_hash": export_hash,
            "view_url": f"http://127.0.0.1:8080/ipfs/{export_hash}"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error exporting results: {str(e)}"}
        )

@app.post("/consolidate-results")
async def consolidate_results(
    server_urls: List[str],
    settings: Settings = Depends(get_settings)
):
    """Fetch and consolidate results from multiple servers"""
    if settings.db_provider != "ipfs":
        return JSONResponse(
            status_code=400,
            content={"error": "This endpoint is only available when using IPFS as the database provider"}
        )
    
    try:
        all_results = []
        errors = []
        
        # Get results from this server
        ipfs_db = IPFSDatabase(settings)
        local_results = ipfs_db.get_recognition_results(limit=1000)
        all_results.extend(local_results)
        
        # Get results from other servers
        for url in server_urls:
            try:
                fetch_url = f"{url.rstrip('/')}/recognition-results?limit=1000"
                response = requests.get(fetch_url, timeout=10)
                
                if response.status_code == 200:
                    server_results = response.json()
                    all_results.extend(server_results)
                else:
                    errors.append(f"Failed to fetch from {url}: {response.status_code}")
            except Exception as e:
                errors.append(f"Error connecting to {url}: {str(e)}")
        
        # Save consolidated results to IPFS
        api_url = "http://127.0.0.1:5001/api/v0/add"
        export_data = {
            "results": all_results,
            "servers": ["local"] + server_urls,
            "count": len(all_results),
            "consolidation_time": datetime.now().isoformat()
        }
        
        json_data = json.dumps(export_data)
        files = {'file': ('consolidated.json', json_data)}
        response = requests.post(api_url, files=files)
        
        if response.status_code != 200:
            raise Exception(f"Failed to add to IPFS: {response.status_code}")
        
        # Parse the response to get the hash
        result = response.json()
        consolidated_hash = result.get('Hash')
        
        return {
            "status": "success" if not errors else "partial_success",
            "result_count": len(all_results),
            "ipfs_hash": consolidated_hash,
            "view_url": f"http://127.0.0.1:8080/ipfs/{consolidated_hash}",
            "errors": errors
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error consolidating results: {str(e)}"}
        )

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