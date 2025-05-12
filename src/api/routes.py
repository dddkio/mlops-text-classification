from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
from pydantic import BaseModel
from .auth import get_current_user
from ..models.factory import ModelFactory
from ..data.preprocessing import TextPreprocessor

router = APIRouter()
model = ModelFactory.get_model('bert')
preprocessor = TextPreprocessor()

class TextRequest(BaseModel):
    text: str

class TextsRequest(BaseModel):
    texts: List[str]

@router.post("/predict")
async def predict_sentiment(
    request: TextRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    try:
        processed_text = preprocessor.clean_text(request.text)
        prediction = model.predict(processed_text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-predict")
async def batch_predict(
    request: TextsRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    try:
        processed_texts = preprocessor.preprocess_data(request.texts)
        predictions = [model.predict(text) for text in processed_texts]
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))