from http.client import HTTPException
import psycopg2
from pgvector.psycopg2 import register_vector
import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()

conn = psycopg2.connect(
    dbname="multimedia_db",
    user="test",
    host="localhost",
    password="123",
    port="5432"
)
register_vector(conn)


@app.get("/hnsw_index")
async def create_index_hnsw():
    """
    Create an hnsw index on the embeddings (vectors) column of the multimedia_embeddings table.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_hnsw 
            ON multimedia_embeddings 
            USING hnsw (embedding vector_l2_ops) 
            WITH (m = 10, ef_construction = 40);
        """)

        conn.commit()
        cursor.close()

        return {"message": "HNSW index created successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
