from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import cohere
import numpy as np
import json
import ssl
import httpx
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import time
import pandas as pd
import urllib3
import io
from datetime import datetime

# Disable SSL warnings when using SSL bypass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Data API",
    description="API for managing healthcare data with vector embeddings",
    version="1.0.0"
)

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for clients
cohere_client = None
config = None

# Pydantic models for API requests/responses
class ConfigResponse(BaseModel):
    status: str
    message: str
    config_loaded: bool

class DataLoadResponse(BaseModel):
    status: str
    message: str
    records_count: int
    columns: List[str]

class EmbeddingResponse(BaseModel):
    status: str
    message: str
    embeddings_count: int
    dimensions: int

class HealthcareRecord(BaseModel):
    id: Optional[int] = None
    Name: str
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Date_of_Admission: str
    Doctor: str
    Hospital: str
    Insurance_Provider: str
    Billing_Amount: float
    Room_Number: str
    Admission_Type: str
    Discharge_Date: str
    Medication: str
    Test_Results: str

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class SearchResult(BaseModel):
    records: List[Dict[str, Any]]
    total_found: int

class ProcessDataResponse(BaseModel):
    status: str
    message: str
    records_count: int
    collection_name: str

class SearchWithChatResponse(BaseModel):
    query: str
    ai_response: str
    retrieved_records: List[Dict[str, Any]]
    total_found: int

# Configuration functions (from original code)
def load_config():
    """Load and validate configuration from .env file"""
    load_dotenv()
    
    config = {
        'cohere_api_key': os.getenv('COHERE_API_KEY'),
        'milvus_uri': os.getenv('MILVUS_URI'),
        'milvus_token': os.getenv('MILVUS_TOKEN'),
        'csv_filename': os.getenv('CSV_FILENAME', 'healthcare_dataset.csv'),
        'batch_size': int(os.getenv('BATCH_SIZE', '100')),
        'ssl_verify': os.getenv('SSL_VERIFY', 'False').lower() == 'true',
        'collection_name': os.getenv('COLLECTION_NAME', 'healthcare_data')
    }
    
    required_vars = ['cohere_api_key', 'milvus_uri', 'milvus_token']
    missing_vars = [var for var in required_vars if not config[var]]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return config

def initialize_cohere_client(api_key, ssl_verify=False):
    """Initialize Cohere client with optional SSL bypass"""
    try:
        if not ssl_verify:
            custom_client = httpx.Client(verify=False)
            co = cohere.Client(api_key=api_key, httpx_client=custom_client)
        else:
            co = cohere.Client(api_key=api_key)
        return co
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing Cohere client: {e}")

def generate_embeddings(texts: List[str], co):
    """Generate embeddings for text data"""
    try:
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {e}")

def connect_to_milvus(uri, token):
    """Connect to Milvus database"""
    try:
        connections.connect(
            alias="default",
            uri=uri,
            token=token
        )
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Milvus: {e}")

def create_collection_schema():
    """Create the collection schema for healthcare data"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="Name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="Age", dtype=DataType.INT64),
        FieldSchema(name="Gender", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="Blood_Type", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="Medical_Condition", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="Date_of_Admission", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="Doctor", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="Hospital", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="Insurance_Provider", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="Billing_Amount", dtype=DataType.FLOAT),
        FieldSchema(name="Room_Number", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="Admission_Type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="Discharge_Date", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="Medication", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="Test_Results", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
    ]
    
    schema = CollectionSchema(fields, description="Healthcare patient records with 1024-d embeddings")
    return schema

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global cohere_client, config
    try:
        config = load_config()
        cohere_client = initialize_cohere_client(config['cohere_api_key'], config['ssl_verify'])
        connect_to_milvus(config['milvus_uri'], config['milvus_token'])
        print("✅ FastAPI app initialized successfully")
    except Exception as e:
        print(f"❌ Error during startup: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Healthcare Data API is running", "timestamp": datetime.now().isoformat()}

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get configuration status"""
    try:
        global config
        if config is None:
            config = load_config()
        
        return ConfigResponse(
            status="success",
            message="Configuration loaded successfully",
            config_loaded=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-csv", response_model=DataLoadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df.fillna("")
        
        return DataLoadResponse(
            status="success",
            message=f"CSV file loaded successfully",
            records_count=len(df),
            columns=list(df.columns)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

def prepare_data_for_insertion(df):
    """Prepare dataframe data for Milvus insertion"""
    try:
        # Handle column name variations
        column_mapping = {
            'Room Number': 'Room_Number',
            'Blood Type': 'Blood_Type',
            'Medical Condition': 'Medical_Condition',
            'Date of Admission': 'Date_of_Admission',
            'Insurance Provider': 'Insurance_Provider',
            'Billing Amount': 'Billing_Amount',
            'Admission Type': 'Admission_Type',
            'Discharge Date': 'Discharge_Date',
            'Test Results': 'Test_Results'
        }
        
        # Convert column names if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
        
        # Ensure required columns exist with defaults
        required_fields = {
            'Name': '',
            'Age': 0,
            'Gender': '',
            'Blood_Type': '',
            'Medical_Condition': '',
            'Date_of_Admission': '',
            'Doctor': '',
            'Hospital': '',
            'Insurance_Provider': '',
            'Billing_Amount': 0.0,
            'Room_Number': '',
            'Admission_Type': '',
            'Discharge_Date': '',
            'Medication': '',
            'Test_Results': ''
        }
        
        for field, default_value in required_fields.items():
            if field not in df.columns:
                df[field] = default_value
        
        # Convert data types
        df['Room_Number'] = df['Room_Number'].astype(str)
        df['Age'] = df['Age'].astype(int)
        df['Billing_Amount'] = df['Billing_Amount'].astype(float)
        
        ids = list(range(1, len(df) + 1))
        
        records = [
            ids,
            df['Name'].tolist(),
            df['Age'].tolist(),
            df['Gender'].tolist(),
            df['Blood_Type'].tolist(),
            df['Medical_Condition'].tolist(),
            df['Date_of_Admission'].tolist(),
            df['Doctor'].tolist(),
            df['Hospital'].tolist(),
            df['Insurance_Provider'].tolist(),
            df['Billing_Amount'].tolist(),
            df['Room_Number'].tolist(),
            df['Admission_Type'].tolist(),
            df['Discharge_Date'].tolist(),
            df['Medication'].tolist(),
            df['Test_Results'].tolist(),
            df['embedding'].tolist()
        ]
        
        return records
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {e}")

def search_milvus_collection(query_emb, limit=10):
    """Search Milvus collection with query embedding"""
    try:
        global config
        collection = Collection(config['collection_name'])
        collection.load()

        results = collection.search(
            data=[query_emb],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=[
                "Name", "Age", "Gender", "Blood_Type", "Medical_Condition",
                "Date_of_Admission", "Doctor", "Hospital", "Insurance_Provider",
                "Billing_Amount", "Room_Number", "Admission_Type", "Discharge_Date",
                "Medication", "Test_Results"
            ]
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching collection: {e}")

def retrieve_from_results(results):
    """Extract and format results from Milvus search"""
    retrieved_records = []
    
    for hit in results[0]:
        record = {
            "id": hit.id,
            "score": hit.score,
            "data": {}
        }
        
        # Extract all fields from the hit
        for field in hit.entity.fields:
            record["data"][field] = hit.entity.get(field)
        
        retrieved_records.append(record)
    
    return retrieved_records

def generate_response_from_context(retrieved_texts, query, co):
    """Generate a response using Cohere Chat with retrieved context"""
    try:
        instructions = (
            "Using the context and documents provided, generate a detailed response that most accurately answers the user's query."
        )

        response = co.chat(
            model='command-r-plus',
            message=query,
            documents=[retrieved_texts],
            prompt_truncation='AUTO',
            temperature=0.7
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

@app.post("/process-data")
async def process_data_endpoint(file: UploadFile = File(...), background_tasks: BackgroundTasks):
    """Process CSV file: generate embeddings and store in Milvus"""
    try:
        global cohere_client, config
        if not cohere_client or not config:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df.fillna("")
        
        # Check if required columns exist
        required_columns = ['Medical Condition', 'Test Results', 'Medication']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Use available columns or create default text
            text_parts = []
            for col in required_columns:
                if col in df.columns:
                    text_parts.append(df[col].astype(str))
                else:
                    text_parts.append(pd.Series([""] * len(df)))
            combined_texts = (text_parts[0] + " | " + text_parts[1] + " | " + text_parts[2]).tolist()
        else:
            # Combine relevant fields for embedding
            combined_texts = (
                df['Medical Condition'].astype(str) + " | " +
                df['Test Results'].astype(str) + " | " +
                df['Medication'].astype(str)
            ).tolist()
        
        # Generate embeddings
        embeddings = generate_embeddings(combined_texts, cohere_client)
        df['embedding'] = embeddings
        
        # Create/recreate collection
        collection_name = config['collection_name']
        
        # Drop existing collection if it exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # Create new collection
        schema = create_collection_schema()
        collection = Collection(name=collection_name, schema=schema)
        
        # Prepare and insert data
        records = prepare_data_for_insertion(df)
        
        # Insert data in batches
        batch_size = config['batch_size']
        total_records = len(records[0])
        
        for i in range(0, total_records, batch_size):
            batch = [field[i:i+batch_size] for field in records]
            collection.insert(batch)

        # Create index and load collection
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        collection.load()
        
        return {
            "status": "success",
            "message": f"Processed {total_records} records successfully",
            "records_count": total_records,
            "collection_name": collection_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResult)
async def search_records(request: SearchRequest):
    """Search healthcare records using vector similarity"""
    try:
        global cohere_client, config
        if not cohere_client or not config:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        # Generate embedding for search query
        query_embedding = generate_embeddings([request.query], cohere_client)[0]
        
        # Search in Milvus
        results = search_milvus_collection(query_embedding, request.limit)
        
        # Extract and format results
        retrieved_records = retrieve_from_results(results)
        
        return SearchResult(
            records=retrieved_records,
            total_found=len(retrieved_records)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-with-chat")
async def search_with_chat(request: SearchRequest):
    """Search records and generate AI response using retrieved context"""
    try:
        global cohere_client, config
        if not cohere_client or not config:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        # Generate embedding for search query
        query_embedding = generate_embeddings([request.query], cohere_client)[0]
        
        # Search in Milvus
        results = search_milvus_collection(query_embedding, request.limit)
        
        # Extract results
        retrieved_records = retrieve_from_results(results)
        
        # Format context for chat
        retrieved_texts = {}
        for record in retrieved_records:
            context_piece = "\n".join([f"{field}: {value}" for field, value in record["data"].items()])
            retrieved_texts[record["id"]] = context_piece
        
        # Generate AI response
        if retrieved_texts:
            ai_response = generate_response_from_context(retrieved_texts, request.query, cohere_client)
        else:
            ai_response = "No relevant healthcare records found for your query."
        
        return {
            "query": request.query,
            "ai_response": ai_response,
            "retrieved_records": retrieved_records,
            "total_found": len(retrieved_records)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List available Milvus collections"""
    try:
        collections = utility.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@app.post("/create-collection")
async def create_collection():
    """Create the healthcare data collection"""
    try:
        global config
        schema = create_collection_schema()
        collection = Collection(config['collection_name'], schema)
        
        # Create index on embedding field
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        return {"message": f"Collection '{config['collection_name']}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
