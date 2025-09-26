from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from data_loader import data_loader
from query_processor import query_processor
from response_generator import response_generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="GNSS Satellite Prediction Chatbot",
    description="AI-powered chatbot for analyzing GNSS satellite prediction performance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatRequest(BaseModel):
    query: str
    format_type: Optional[str] = "text"  # "text" or "json"

class ChatResponse(BaseModel):
    success: bool
    response_type: str
    text: Optional[str] = None
    data: Dict[str, Any]
    summary: Optional[str] = None

# Global variable to track initialization
_initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize data loading on startup"""
    global _initialized
    
    logger.info("Starting GNSS Satellite Prediction Chatbot...")
    
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY environment variable not set!")
        raise Exception("GOOGLE_API_KEY is required")
    
    # Load data
    logger.info("Loading satellite data...")
    success = data_loader.load_data()
    
    if not success:
        logger.error("Failed to load data files!")
        raise Exception("Failed to load required data files")
    
    # Log data summary
    summary = data_loader.get_data_summary()
    logger.info(f"Data loaded successfully: {summary}")
    
    _initialized = True
    logger.info("Chatbot initialization complete!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    if not _initialized:
        raise HTTPException(status_code=503, detail="Chatbot still initializing")
    
    summary = data_loader.get_data_summary()
    return {
        "message": "GNSS Satellite Prediction Chatbot API",
        "status": "ready",
        "data_summary": summary,
        "endpoints": {
            "/chat": "POST - Main chatbot endpoint",
            "/health": "GET - Health check",
            "/": "GET - API information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not _initialized:
        return {
            "status": "initializing",
            "data_loaded": False
        }
    
    try:
        # Quick data access test
        satellites = data_loader.get_available_satellites()
        return {
            "status": "healthy",
            "data_loaded": True,
            "satellites_count": len(satellites),
            "google_api_configured": bool(os.getenv("GOOGLE_API_KEY"))
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chatbot endpoint"""
    if not _initialized:
        raise HTTPException(status_code=503, detail="Chatbot still initializing")
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Validate format type
        if request.format_type not in ["text", "json"]:
            raise HTTPException(status_code=400, detail="format_type must be 'text' or 'json'")
        
        # Process the query
        analysis_result = query_processor.process_query(request.query)
        
        # Generate formatted response
        formatted_response = response_generator.generate_response(
            analysis_result, 
            request.format_type
        )
        
        # Return response
        return ChatResponse(
            success=True,
            response_type=formatted_response.get("response_type", "unknown"),
            text=formatted_response.get("text"),
            data=formatted_response.get("data", {}),
            summary=formatted_response.get("summary")
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return ChatResponse(
            success=False,
            response_type="error",
            text=f"Sorry, I encountered an error: {str(e)}",
            data={"error": str(e)},
            summary="Error processing request"
        )

# Example usage endpoint for testing
@app.get("/satellites")
async def get_available_satellites():
    """Get list of available satellites"""
    if not _initialized:
        raise HTTPException(status_code=503, detail="Chatbot still initializing")
    
    try:
        satellites = data_loader.get_available_satellites()
        return {
            "satellites": satellites,
            "count": len(satellites),
            "sample_queries": [
                f"How does {satellites[0]} perform?",
                f"Compare {satellites[0]} and {satellites[1]}",
                f"Show statistics for {satellites[0]}"
            ] if len(satellites) >= 2 else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples")
async def get_example_queries():
    """Get example queries for testing"""
    return {
        "example_queries": [
            "Compare satellites G01 and G14",
            "Which satellite has the best performance?",
            "Show me statistics for satellite G01",
            "Find satellites with error below 5%",
            "How does G01 perform across different prediction horizons?",
            "Which satellite between G01, G14, and G20 performs best at 2-hour predictions?",
            "Show me error analysis for G07",
            "Compare all GPS satellites",
            "What satellites have RMSE below 0.05?"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)