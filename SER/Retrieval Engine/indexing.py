from http.client import HTTPException
import psycopg2
from pgvector.psycopg2 import register_vector
import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()


# connection to the local database
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
    TODO: change to the correct parameters
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE INDEX idx_embeddings_hnsw ON embeddings 
            USING hnsw (content_vector vector_l2_ops) WITH (m = 10, ef_construction = 40);
        """)
        cursor.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/ivfflat_index")
async def create_index_ivfflat():
    """
    Create an hnsw index on the embeddings (vectors) column of the multimedia_embeddings table.
    TODO: change to the correct parameters
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE INDEX idx_embeddings_ivfflat ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
        """)
        cursor.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
