from importlib import reload
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# CORS middleware to allow requests from frontend
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None
model_info = {}

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    label: str
    score: float
    confidence_percentage: float


def load_model():
    global classifier, model_info

    try:
        model_path = "model"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model directory '{model_path}' does not exist.")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

        model_info = {
            "model_name": model_path,
            "num_labels": 2,
            "labels": ["Negative", "Positive"]
        }

        logger.info("Model loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed.")
        return False
    
@app.on_event("startup")
def startup_event():
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup.")
        raise HTTPException(status_code=500, detail="Model loading failed during startup.")
    

@app.get("/health")
async def health_check():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Service unavailable. Model not loaded.")
    return {"status": "healthy", "model_info": model_info}  

@app.post("/predict", response_model=PredictionResponse)
async def predict(input: TextInput):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Service unavailable. Model not loaded.")
    
    try:
        results = classifier(input.text)

        prediction = results[0]
        label = prediction['label']
        score = prediction['score']
        confidence_percentage = score * 100

        return PredictionResponse(
            text=input.text,
            label=label,
            score=score,
            confidence_percentage=confidence_percentage
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)