from fastapi import FastAPI
from src.modules.user.userController import userRouter
from src.modules.analysis.analysisController import analysisRouter
from src.modules.user.userModel import database

app = FastAPI()

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

app.include_router(userRouter)
app.include_router(analysisRouter)
