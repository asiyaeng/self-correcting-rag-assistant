# FastAPI Reference Guide

FastAPI is a modern, fast web framework for building APIs with Python based on standard type hints.

## Installation
```bash
pip install fastapi uvicorn[standard]
```

## Quick Start
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```

Run with:
```bash
uvicorn main:app --reload
```

## Path Operations

### HTTP Methods
```python
@app.get("/items/{item_id}")
@app.post("/items/")
@app.put("/items/{item_id}")
@app.delete("/items/{item_id}")
@app.patch("/items/{item_id}")
```

### Path Parameters
```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}
```

### Query Parameters
```python
@app.get("/items/")
async def list_items(skip: int = 0, limit: int = 10, search: str | None = None):
    return {"skip": skip, "limit": limit, "search": search}
```

## Request Bodies

### Pydantic Models
```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str
    price: float = Field(gt=0, description="Must be positive")
    tags: list[str] = []

@app.post("/items/", status_code=201)
async def create_item(item: Item) -> Item:
    return item
```

### File Uploads
```python
from fastapi import File, UploadFile

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}
```

## Response Models
```python
from pydantic import BaseModel

class UserOut(BaseModel):
    id: int
    username: str
    # password field is intentionally excluded

@app.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: int) -> UserOut:
    # Even if the returned dict has a "password" key, it won't appear in the response
    return UserOut(id=user_id, username="alice")
```

## HTTP Status Codes
```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return item
```

Common status codes:
- `200 OK` — default for GET/PUT
- `201 Created` — successful POST that creates a resource
- `204 No Content` — successful DELETE
- `400 Bad Request` — invalid input
- `404 Not Found` — resource not found
- `422 Unprocessable Entity` — validation error (FastAPI default for Pydantic errors)
- `500 Internal Server Error` — unexpected server error

## Error Handling

### HTTPException
```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in db:
        raise HTTPException(status_code=404, detail="Item not found")
    return db[item_id]
```

### Custom Exception Handlers
```python
from fastapi import Request
from fastapi.responses import JSONResponse

class CustomError(Exception):
    def __init__(self, message: str):
        self.message = message

@app.exception_handler(CustomError)
async def custom_error_handler(request: Request, exc: CustomError):
    return JSONResponse(status_code=400, content={"detail": exc.message})
```

## Dependency Injection
```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/items/")
async def list_items(db = Depends(get_db)):
    return db.query(Item).all()
```

## Background Tasks
```python
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    # Runs after the response is sent
    print(f"Sending email to {email}: {message}")

@app.post("/send-notification/")
async def notify(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email, email, "Hello!")
    return {"message": "Notification scheduled"}
```

## Middleware
```python
from fastapi.middleware.cors import CORSMiddleware
import time

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start)
    return response
```

## Routers (APIRouter)
Organise routes into separate modules:

```python
# routers/items.py
from fastapi import APIRouter

router = APIRouter(prefix="/items", tags=["Items"])

@router.get("/")
async def list_items():
    return []

# main.py
from routers import items
app.include_router(items.router)
```

## Lifespan Events
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
```

## OpenAPI / Swagger Docs
FastAPI auto-generates interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

Customise docs:
```python
app = FastAPI(
    title="My API",
    description="API description with **markdown** support",
    version="2.0.0",
    docs_url="/api-docs",
)
```

## Testing with TestClient
```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
```

## Async vs Sync
- Use `async def` for I/O-bound operations (HTTP calls, DB queries, file reads)
- Use regular `def` for CPU-bound operations (FastAPI runs them in a thread pool)
- Mixing both in the same app is safe — FastAPI handles it automatically

## Common Troubleshooting

### 422 Validation Error
Pydantic validation failed. Check the error detail in the response body — it lists exactly which fields failed and why.

### CORS Errors
Add `CORSMiddleware` and ensure `allow_origins` includes your frontend URL.

### Startup errors with lifespan
If lifespan fails, the app won't start. Wrap startup code in try/except to log errors without crashing.
