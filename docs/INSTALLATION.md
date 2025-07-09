# Calquity Installation Guide

**Author:** Divyanshu Chaudhary  
**Version:** 1.0.0  

## Prerequisites

- Python 3.8 or higher
- PostgreSQL 14+ with PGVector extension
- Git (for cloning repository)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/divyanshu-chaudhary/calquity.git
cd calquity
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r config/requirements.txt
```

### 4. Database Setup

```sql
-- Create database
CREATE DATABASE calquitytask;

-- Install PGVector extension
CREATE EXTENSION vector;

-- Create documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    company VARCHAR(10),
    document_type VARCHAR(50),
    year INTEGER,
    quarter VARCHAR(10),
    content TEXT,
    dense_embedding vector(384)
);
```

### 5. Environment Configuration

Create `.env` file:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=calquitytask
DB_USER=postgres
DB_PASSWORD=your_password
```

### 6. Run Tests

```bash
python tests/test_system.py
```

## Quick Start

```python
from calquity import HybridSearchEngine

# Initialize search engine
engine = HybridSearchEngine()

# Perform search
results = engine.search("revenue growth strategy", limit=10)

# Display results
for result in results:
    print(f"{result['company']}: {result['content_preview'][:100]}...")
```
