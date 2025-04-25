import MemePy
from fastapi import FastAPI
import uvicorn
import pyjokes
from compliment import Compliment

app = FastAPI()


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
        c = Compliment()
        compliment = c.make_compliment()
        return {"compliment": compliment}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
