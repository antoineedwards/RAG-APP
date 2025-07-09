from typing import Any, List, Dict, Optional
import uuid
import logging
from dataclasses import dataclass

from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from backend.config.settings import Settings
from backend.schemas.context import Context
from backend.schemas.tool import ToolCategory, ToolDefinition
from backend.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


class MilvusVectorStore:
    """
    Milvus/Zilliz vector store implementation for document storage and retrieval.
    """
    
    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        secure: bool = False,
        embedding_dim: int = 4096,  # Cohere embedding dimension
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.secure = secure
        self.embedding_dim = embedding_dim
        self.collection = None
        
    def connect(self):
        """Connect to Milvus/Zilliz"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                secure=self.secure,
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def create_collection(self):
        """Create collection with schema if it doesn't exist"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            return
        
        # Define schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=128,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=32768,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="page_number",
                dtype=DataType.INT64,
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Collection for RAG documents: {self.collection_name}",
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default',
            shards_num=2,
        )
        
        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        
        logger.info(f"Created collection: {self.collection_name}")
    
    def insert_documents(self, documents: List[DocumentChunk]) -> List[str]:
        """Insert documents into the collection"""
        if not documents:
            return []
        
        # Prepare data for insertion
        data = [
            [doc.id for doc in documents],  # ids
            [doc.text for doc in documents],  # texts
            [doc.embedding for doc in documents],  # embeddings
            [doc.metadata.get("source", "") for doc in documents],  # sources
            [doc.metadata.get("title", "") for doc in documents],  # titles
            [doc.metadata.get("page_number", 0) for doc in documents],  # page_numbers
        ]
        
        # Insert data
        mr = self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"Inserted {len(documents)} documents into {self.collection_name}")
        return mr.primary_keys
    
    def search_documents(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Load collection
        self.collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=None,
            output_fields=["text", "source", "title", "page_number"],
        )
        
        # Format results
        formatted_results = []
        for hit in results[0]:
            if hit.score >= score_threshold:
                formatted_results.append({
                    "id": hit.id,
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "title": hit.entity.get("title"),
                    "page_number": hit.entity.get("page_number"),
                    "score": hit.score,
                })
        
        return formatted_results
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        if not document_ids:
            return True
        
        # Create expression for deletion
        ids_str = "', '".join(document_ids)
        expr = f"id in ['{ids_str}']"
        
        # Delete documents
        self.collection.delete(expr)
        self.collection.flush()
        
        logger.info(f"Deleted {len(document_ids)} documents from {self.collection_name}")
        return True
    
    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            return {}
        
        stats = self.collection.num_entities
        return {
            "collection_name": self.collection_name,
            "num_entities": stats,
            "schema": self.collection.schema.to_dict(),
        }


class MilvusVectorDBRetriever(BaseTool):
    """
    This class retrieves documents from a Milvus/Zilliz vector database.
    """
    ID = "milvus_retriever"
    COHERE_API_KEY = Settings().get('deployments.cohere_platform.api_key')
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        secure: bool = False,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.secure = secure
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.embeddings = None
        
        if self.COHERE_API_KEY:
            self.embeddings = CohereEmbeddings(cohere_api_key=self.COHERE_API_KEY)
    
    @classmethod
    def is_available(cls) -> bool:
        return cls.COHERE_API_KEY is not None
    
    @classmethod
    def get_tool_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name=cls.ID,
            display_name="Milvus Vector DB Retriever",
            implementation=cls,
            parameter_definitions={
                "query": {
                    "description": "Query for retrieval.",
                    "type": "str",
                    "required": True,
                },
                "limit": {
                    "description": "Maximum number of documents to retrieve.",
                    "type": "int",
                    "required": False,
                    "default": 10,
                },
                "score_threshold": {
                    "description": "Minimum similarity score threshold.",
                    "type": "float",
                    "required": False,
                    "default": 0.7,
                },
            },
            is_visible=True,
            is_available=cls.is_available(),
            error_message=cls.generate_error_message(),
            category=ToolCategory.DataLoader,
            description="Retrieves documents from Milvus/Zilliz vector database.",
        )
    
    def _initialize_vector_store(self):
        """Initialize vector store connection"""
        if not self.vector_store:
            self.vector_store = MilvusVectorStore(
                collection_name=self.collection_name,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                secure=self.secure,
            )
            self.vector_store.connect()
            self.vector_store.create_collection()
    
    async def call(
        self, parameters: dict, ctx: Context, **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search for relevant documents"""
        if not self.embeddings:
            return self.get_tool_error(details="Cohere API key not configured")
        
        query = parameters.get("query", "")
        limit = parameters.get("limit", 10)
        score_threshold = parameters.get("score_threshold", 0.7)
        
        if not query:
            return self.get_tool_error(details="Query parameter is required")
        
        try:
            # Initialize vector store
            self._initialize_vector_store()
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search for similar documents
            results = self.vector_store.search_documents(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
            )
            
            if not results:
                return self.get_no_results_error()
            
            # Format results for the toolkit
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result["text"],
                    "title": result.get("title", ""),
                    "url": result.get("source", ""),
                    "page_number": result.get("page_number", 0),
                    "score": result["score"],
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in Milvus retrieval: {e}")
            return self.get_tool_error(details=str(e))
    
    def insert_pdf_documents(self, filepath: str) -> bool:
        """Insert PDF documents into the vector store"""
        if not self.embeddings:
            logger.error("Cohere API key not configured")
            return False
        
        try:
            # Initialize vector store
            self._initialize_vector_store()
            
            # Load and split PDF
            loader = PyPDFLoader(filepath)
            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            pages = loader.load_and_split(text_splitter)
            
            # Create document chunks
            document_chunks = []
            for page in pages:
                doc_id = str(uuid.uuid4())
                embedding = self.embeddings.embed_query(page.page_content)
                
                chunk = DocumentChunk(
                    id=doc_id,
                    text=page.page_content,
                    embedding=embedding,
                    metadata={
                        "source": filepath,
                        "title": page.metadata.get("title", ""),
                        "page_number": page.metadata.get("page", 0),
                    },
                )
                document_chunks.append(chunk)
            
            # Insert documents
            inserted_ids = self.vector_store.insert_documents(document_chunks)
            logger.info(f"Inserted {len(inserted_ids)} documents from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting PDF documents: {e}")
            return False
    
    def delete_documents_by_source(self, source: str) -> bool:
        """Delete all documents from a specific source"""
        try:
            # Initialize vector store
            self._initialize_vector_store()
            
            # Search for documents from the source
            # Note: This is a simplified approach. In practice, you might need
            # to implement a more sophisticated query mechanism
            self.vector_store.collection.load()
            
            # Delete by expression
            expr = f'source == "{source}"'
            self.vector_store.collection.delete(expr)
            self.vector_store.collection.flush()
            
            logger.info(f"Deleted documents from source: {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            self._initialize_vector_store()
            return self.vector_store.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Example usage and utility functions
class MilvusRAGManager:
    """
    High-level manager for RAG operations with Milvus
    """
    
    def __init__(self, retriever: MilvusVectorDBRetriever):
        self.retriever = retriever
    
    def add_pdf_to_knowledge_base(self, pdf_path: str) -> bool:
        """Add a PDF to the knowledge base"""
        return self.retriever.insert_pdf_documents(pdf_path)
    
    def remove_pdf_from_knowledge_base(self, pdf_path: str) -> bool:
        """Remove a PDF from the knowledge base"""
        return self.retriever.delete_documents_by_source(pdf_path)
    
    async def search_knowledge_base(
        self, query: str, limit: int = 10, score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base"""
        parameters = {
            "query": query,
            "limit": limit,
            "score_threshold": score_threshold,
        }
        
        # Create a mock context (you'll need to adapt this to your actual Context class)
        ctx = Context()  # Replace with your actual context creation
        
        results = await self.retriever.call(parameters, ctx)
        return results
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return self.retriever.get_stats()


# Example configuration for different environments
def create_local_milvus_retriever() -> MilvusVectorDBRetriever:
    """Create retriever for local Milvus instance"""
    return MilvusVectorDBRetriever(
        collection_name="local_rag_docs",
        host="localhost",
        port="19530",
        secure=False,
    )


def create_zilliz_cloud_retriever(
    endpoint: str, token: str, collection_name: str = "cloud_rag_docs"
) -> MilvusVectorDBRetriever:
    """Create retriever for Zilliz Cloud"""
    # Extract host and port from Zilliz Cloud endpoint
    # Format: https://your-cluster-endpoint.zillizcloud.com:443
    host = endpoint.replace("https://", "").replace("http://", "").split(":")[0]
    port = "443" if "443" in endpoint else "19530"
    
    return MilvusVectorDBRetriever(
        collection_name=collection_name,
        host=host,
        port=port,
        user="",  # Zilliz Cloud uses token authentication
        password=token,
        secure=True,
    )
