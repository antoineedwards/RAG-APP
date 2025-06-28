Key Features:

FastAPI Structure: Full REST API with proper error handling and documentation
CORS Middleware: Configured for Next.js frontend (ports 3000)
Pydantic Models: Type-safe request/response models
Configuration Management: Loads from .env file on startup
File Upload: Endpoint to upload CSV files
Database Connections: Initializes Cohere and Milvus connections

Complete Data Processing Pipeline:

POST /process-data: Handles the entire pipeline - uploads CSV, generates embeddings, creates Milvus collection, and inserts data
Data preparation: Handles column mapping, data type conversion, and missing fields
Batch processing: Inserts data in configurable batches for efficiency

Advanced Search Capabilities:

POST /search: Vector similarity search returning raw results
POST /search-with-chat: Search + AI-powered response generation using Cohere's chat model
Flexible search parameters: Configurable result limits

Complete Milvus Integration:

Collection creation with proper schema
Index creation for optimal search performance
Batch data insertion
Vector similarity search with IP (Inner Product) metric

![image](https://github.com/user-attachments/assets/0e383d94-84d9-4e39-89b0-2a03209d1b1d)
