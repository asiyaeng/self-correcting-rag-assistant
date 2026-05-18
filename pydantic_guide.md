# Pydantic v2 — Data Validation Guide

Pydantic is a data validation library using Python type hints. FastAPI uses Pydantic for request/response validation.

## Basic Models

```python
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class User(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime = Field(default_factory=datetime.now)
    age: Optional[int] = None

user = User(id=1, name="Alice", email="alice@example.com")
print(user.model_dump())  # {'id': 1, 'name': 'Alice', ...}
```

## Field Validation

```python
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="Must be positive")
    tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        return v.strip()
```

## Model Config

```python
from pydantic import BaseModel, ConfigDict

class APIModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",          # ignore unknown fields
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,             # immutable instances
    )
```

## Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str = "US"

class Person(BaseModel):
    name: str
    address: Address

person = Person(name="Bob", address={"street": "123 Main St", "city": "NYC"})
```

## Enums with Pydantic

```python
from enum import Enum
from pydantic import BaseModel

class Status(str, Enum):
    active = "active"
    inactive = "inactive"
    pending = "pending"

class Task(BaseModel):
    title: str
    status: Status = Status.pending
```

## Serialization

```python
user = User(id=1, name="Alice", email="alice@example.com")

# To dict
d = user.model_dump()
d_json_safe = user.model_dump(mode="json")  # converts datetime to ISO string

# To JSON string
json_str = user.model_dump_json()

# Exclude/include fields
d = user.model_dump(exclude={"email"})
d = user.model_dump(include={"id", "name"})
```

## Parsing / Validation

```python
# From dict
user = User.model_validate({"id": 1, "name": "Alice", "email": "a@b.com"})

# From JSON string
user = User.model_validate_json('{"id": 1, "name": "Alice", "email": "a@b.com"}')
```

## Settings Management (pydantic-settings)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str
    api_key: str = ""
    debug: bool = False
    max_connections: int = 10

settings = Settings()  # reads from environment + .env file
```

## Common Validation Patterns

### Email Validation
```python
from pydantic import EmailStr  # requires: pip install pydantic[email]

class User(BaseModel):
    email: EmailStr
```

### URL Validation
```python
from pydantic import HttpUrl

class Config(BaseModel):
    webhook_url: HttpUrl
```

### Custom Validators
```python
from pydantic import model_validator

class DateRange(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def check_dates(self) -> "DateRange":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self
```

## Error Handling

```python
from pydantic import ValidationError

try:
    User(id="not-an-int", name="Alice", email="bad-email")
except ValidationError as e:
    print(e.error_count())  # 2
    for error in e.errors():
        print(error["loc"], error["msg"], error["type"])
```

## Pydantic with FastAPI
FastAPI automatically uses Pydantic for:
- Request body validation
- Query/path parameter validation
- Response model serialization
- OpenAPI schema generation

```python
@app.post("/users/", response_model=UserOut, status_code=201)
async def create_user(user: UserIn) -> UserOut:
    # user is already validated by Pydantic
    return UserOut(**user.model_dump())
```

## Performance Tips
- Use `model_config = ConfigDict(frozen=True)` for read-only models (enables hashing)
- Prefer `model_validate` over `__init__` for bulk construction from dicts
- Use `model_construct` (no validation) only when data is pre-validated for speed
