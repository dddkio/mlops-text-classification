from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from .routes import router as api_router
from .auth import get_current_user

app = FastAPI(title="Text Classification API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    api_router,
    prefix="/api/v1",
    dependencies=[Depends(get_current_user)]
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)