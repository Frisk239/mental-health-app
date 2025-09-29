from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import emotion, farm, social, history, debug, emotion_ws

# Create FastAPI app
app = FastAPI(
    title="Mental Health Assistant API",
    description="AI-powered mental health support system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(emotion.router, prefix="/api/emotion", tags=["emotion"])
app.include_router(farm.router, prefix="/api/farm", tags=["farm"])
app.include_router(social.router, prefix="/api/social", tags=["social"])
app.include_router(history.router, prefix="/api/history", tags=["history"])
app.include_router(debug.router, tags=["debug"])
app.include_router(emotion_ws.router, prefix="/api/emotion", tags=["emotion_ws"])

@app.get("/")
async def root():
    return {"message": "Mental Health Assistant API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
