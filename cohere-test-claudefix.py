import os
from dotenv import load_dotenv
import cohere
import numpy as np
import json
import ssl
import httpx
from typing import List
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import time
import pandas as pd
import urllib3

# Disable SSL warnings when using SSL bypass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_config():
    """Load and validate configuration from .env file"""
    # Load environment variables from .env file
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
    
    # Validate required variables
    required_vars = ['cohere_api_key', 'milvus_uri', 'milvus_token']
    missing_vars = [var for var in required_vars if not config[var]]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("✅ Configuration loaded from .env file")
    return config

def initialize_cohere_client(api_key, ssl_verify=False):
    """Initialize Cohere client with optional SSL bypass"""
    try:
        if not ssl_verify:
            # Create custom HTTP client that bypasses SSL verification
            custom_client = httpx.Client(verify=False)
            co = cohere.Client(api_key=api_key, httpx_client=custom_client)
            print("🔓 Cohere client initialized with SSL bypass")
        else:
            co = cohere.Client(api_key=api_key)
            print("🔒 Cohere client initialized with SSL verification")
        
        return co
        
    except Exception as e:
        print(f"❌ Error initializing Cohere client: {e}")
        return None

def load_and_prepare_data(filename):
    """Load and prepare healthcare data"""
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"CSV file '{filename}' not found")
            
        df = pd.read_csv(filename)
        df = df.fillna("")
        print(f"✅ Loaded {len(df)} records from {filename}")
        
        # Display column names for debugging
        print(f"📋 CSV columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def generate_embeddings(df, co):
    """Generate embeddings for the healthcare data"""
    try:
        # Check if required columns exist
        required_columns = ['Medical Condition', 'Test Results', 'Medication']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            print("Available columns:", list(df.columns))
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
        
        print(f"🔄 Generating embeddings for {len(combined_texts)} records...")
        
        # Generate embeddings with error handling
        response = co.embed(
            texts=combined_texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        
        embeddings = response.embeddings
        print(f"✅ Generated {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")
        return embeddings
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        print(f"Error type: {type(e).__name__}")
        return None

def connect_to_milvus(uri, token):
    """Connect to Milvus database"""
    try:
        connections.connect(
            alias="default",
            uri=uri,
            token=token
        )
        print("✅ Connected to Milvus successfully")
        return True
    except Exception as e:
        print(f"❌ Error connecting to Milvus: {e}")
        return False

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
                print(f"⚠️ Added missing field '{field}' with default value")
        
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
        
        print("✅ Data prepared for insertion")
        return records
        
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return None

def generate_query_embedding(co):
    query = input("Enter your query")

    query_emb = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    return [query, query_emb]

def search_milvus_collection(query_emb):
    collection = Collection("healthcare_data")
    collection.load()

    results = collection.search(
        data=[query_emb],  # Wrap in list for batch search
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=100,
        output_fields=[
            "Name", "Age", "Gender", "Blood_Type", "Medical_Condition",
            "Date_of_Admission", "Doctor", "Hospital", "Insurance_Provider",
            "Billing_Amount", "Room_Number", "Admission_Type", "Discharge_Date",
            "Medication", "Test_Results"
        ]
    )
    return results

def retrieve_from_results(results):
    retrieved_texts = {}
    for hit in results[0]:
        fields = hit.entity.fields
        context_piece = "\n".join([f"{field}: {hit.entity.get(field)}" for field in fields])
        retrieved_texts[hit.id] = context_piece
    
    return retrieved_texts

def generate_response_from_context(retrieved_texts, query, co):
# Generate a response using Cohere Chat
    instructions = (
        "Using the context and documents provided, generate a detailed response that most accurately answers the user's query. "
    )

    response = co.chat(
        model='command-r-plus',
        message=query,
        documents=[retrieved_texts],
        prompt_truncation='AUTO',
        temperature=0.7
    )
    print("Answer:\n", response.text)

def main():
    
    """Main execution function"""
    print("🚀 Starting healthcare vector database setup...")
    
    try:
        # Step 1: Load configuration
        config = load_config()
        
        # Step 2: Initialize Cohere client
        co = initialize_cohere_client(config['cohere_api_key'], config['ssl_verify'])
        if co is None:
            return
    
        # Step 3: Load data
        df = load_and_prepare_data(config['csv_filename'])
        if df is None:
            return
        
        # Step 4: Generate embeddings
        embeddings = generate_embeddings(df, co)
        if embeddings is None:
            return
        
        df['embedding'] = embeddings
        
        # Step 5: Connect to Milvus
        if not connect_to_milvus(config['milvus_uri'], config['milvus_token']):
            return
        
        # Step 6: Create/recreate collection
        collection_name = config['collection_name']
        
        # Drop existing collection if it exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"🗑️ Dropped existing collection: {collection_name}")
        
        # Create new collection
        schema = create_collection_schema()
        collection = Collection(name=collection_name, schema=schema)
        print(f"✅ Created collection: {collection_name}")
        
        # Step 7: Prepare and insert data
        print("🔄 Preparing data for insertion...")
        records = prepare_data_for_insertion(df)
        if records is None:
            return
        
        # Insert data in batches
        batch_size = config['batch_size']
        total_records = len(records[0])  # Use first field (ids) to get count
        total_batches = (total_records + batch_size - 1) // batch_size
        
        print(f"📥 Inserting {total_records} records in {total_batches} batches...")
        
        for i in range(0, total_records, batch_size):
            batch = [field[i:i+batch_size] for field in records]
            collection.insert(batch)
            print(f"📥 Inserted batch {(i//batch_size)+1}/{total_batches}")

        # Step 8: Create index and load collection
        print("🔄 Creating index...")
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        
        print("🔄 Loading collection...")
        collection.load()
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")

    query_emb = generate_query_embedding(co)

    results = search_milvus_collection(query_emb[1])

    retrieved_texts = retrieve_from_results(results)
    if retrieved_texts:
        generate_response_from_context(retrieved_texts, query_emb[0], co)
    else:
        print("No relevant texts found.")

if __name__ == "__main__":
    main()