import os from backend.server import app from dotenv import load_dotenv

load_dotenv()

if **name** == "**main**": import uvicorn port =
int(os.environ.get("PORT", 8000)) uvicorn.run(app, host="0.0.0.0",
port=port)