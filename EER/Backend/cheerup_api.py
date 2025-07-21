from fastapi import FastAPI
import uvicorn
import pyjokes
import random

app = FastAPI()

compliments = [
    "You have such nice hair!",
    "You have a great smile!",
    "You light up the world!",
    "You make the world a better place just by being in it!",
    "Your kindness is contagious!",
    "You have great sense of humour!"
]

@app.get("/joke")
async def get_joke():
    try:
        joke = pyjokes.get_joke("en")
        return {"joke": joke}
    except Exception as e:
        return {"error": str(e)}


@app.get("/compliment")
async def get_joke():
    try:
        return {"compliment": random.choice(compliments)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)