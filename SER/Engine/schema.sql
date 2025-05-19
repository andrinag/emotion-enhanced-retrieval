CREATE TABLE multimedia_objects (
object_id SERIAL PRIMARY KEY,
location VARCHAR(255)
);

CREATE TABLE multimedia_embeddings (
id SERIAL PRIMARY KEY,
object_id INTEGER,
frame INTEGER,
frame_time FLOAT,
frame_location VARCHAR(255),
embedding vector(512),
FOREIGN KEY(object_id) REFERENCES multimedia_objects(object_id)
);

CREATE TABLE OCR (
id SERIAL PRIMARY KEY,
embedding_id INTEGER,
text VARCHAR(50000),
emotion VARCHAR(30),
ocr_confidence FLOAT,
sentiment_confidence FLOAT,
sentiment VARCHAR(30),
path_annotated_location VARCHAR(255),
x FLOAT,
y FLOAT,
w FLOAT,
h FLOAT,
FOREIGN KEY (embedding_id) REFERENCES multimedia_embeddings(id)
);

CREATE TABLE Face (
id SERIAL PRIMARY KEY,
embedding_id INTEGER,
emotion VARCHAR(30),
confidence FLOAT,
sentiment VARCHAR(30),
path_annotated_faces VARCHAR(255),
FOREIGN KEY (embedding_id) REFERENCES multimedia_embeddings(id)
);

CREATE TABLE ASR (
id SERIAL PRIMARY KEY,
embedding_id INTEGER,
text VARCHAR(50000),
emotion VARCHAR(30),
confidence FLOAT,
sentiment VARCHAR(30),
FOREIGN KEY (embedding_id) REFERENCES multimedia_embeddings(id)
);