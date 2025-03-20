CREATE TABLE multimedia_objects (
object_id SERIAL PRIMARY KEY,
location VARCHAR(255)
);

CREATE TABLE multimedia_embeddings (
id SERIAL PRIMARY KEY,
object_id INTEGER,
frame INTEGER,
frame_time FLOAT,
embedding vector(512)
);

CREATE TABLE OCR (
id SERIAL PRIMARY KEY,
emotion VARCHAR(30),
confidence FLOAT,
sentiment VARCHAR(30),
path_annotated_location VARCHAR(255),
y_axis1 FLOAT,
x_axis1 FLOAT,
y_axis2 FLOAT,
x_axis2 FLOAT
)

CREATE TABLE Face (
id SERIAL PRIMARY KEY,
emotion VARCHAR(30),
confidence FLOAT,
sentiment VARCHAR(30),
path_annotated_faces VARCHAR(255)
)

CREATE TABLE ASR (
id SERIAL PRIMARY KEY,
text VARCHAR(50000),
emotion VARCHAR(30),
confidence FLOAT,
sentiment VARCHAR(30)
)