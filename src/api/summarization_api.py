"""
FastAPI service for chat summarization
Provides REST API endpoint for generating summaries
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import sys
from typing import Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chat Summarization API",
    description="AI-powered API for summarizing customer support conversations",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Request/Response models
class SummarizeRequest(BaseModel):
    """Request model for summarization endpoint"""
    conversation: str = Field(
        ..., 
        description="The conversation text to summarize",
        min_length=10,
        max_length=5000,
        examples=["Amanda: Hi! Jerry: Hello! Amanda: How are you?"]
    )
    max_length: Optional[int] = Field(
        128,
        description="Maximum length of generated summary",
        ge=20,
        le=256
    )
    num_beams: Optional[int] = Field(
        4,
        description="Number of beams for beam search",
        ge=1,
        le=10
    )

class SummarizeResponse(BaseModel):
    """Response model for summarization endpoint"""
    summary: str = Field(..., description="Generated summary of the conversation")
    conversation_length: int = Field(..., description="Number of words in original conversation")
    summary_length: int = Field(..., description="Number of words in generated summary")
    compression_ratio: float = Field(..., description="Ratio of original to summary length")

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    model_loaded: bool
    model_path: str

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup"""
    global model, tokenizer
    
    logger.info("="*60)
    logger.info("Loading summarization model...")
    logger.info("="*60)
    
    model_path = "models/best_model"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        logger.info(f"✅ Model loaded successfully on {device}")
        logger.info(f"Model path: {model_path}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Chat Summarization API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "summarize": "/summarize",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path="models/best_model"
    )

@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize_conversation(request: SummarizeRequest):
    """
    Generate a summary for a conversation
    
    Args:
        request: SummarizeRequest containing conversation text and parameters
        
    Returns:
        SummarizeResponse with generated summary and metrics
    """
    if model is None or tokenizer is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        logger.info(f"Received summarization request for conversation of length {len(request.conversation)}")
        
        # Tokenize input
        inputs = tokenizer(
            request.conversation,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=request.max_length,
                num_beams=request.num_beams,
                length_penalty=2.0,
                early_stopping=True
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Calculate metrics
        conv_words = len(request.conversation.split())
        summary_words = len(summary.split())
        compression_ratio = conv_words / summary_words if summary_words > 0 else 0
        
        logger.info(f"Generated summary: {summary[:100]}...")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        
        return SummarizeResponse(
            summary=summary,
            conversation_length=conv_words,
            summary_length=summary_words,
            compression_ratio=round(compression_ratio, 2)
        )
        
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "src.api.summarization_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )