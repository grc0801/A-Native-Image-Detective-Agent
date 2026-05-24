from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# templates folder
templates = Jinja2Templates(directory="templates")

# static folder (css)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    context = {
        "request": request,
        "name": "Giridharan B",
        "title": "Engineering Student Portfolio",

        "about": """
        Final year engineering student showcasing academic projects,
        technical explorations, and practical implementations developed
        during the learning journey.
        """,

        "projects": [
            {
                "name": "Smart Text and Speech Assistant",
                "desc": "Speech recognition and NLP based assistant using deep learning models.",
                "tech": "Python, NLP, Deep Learning"
            },
            {
                "name": "Sign Language Interpreter",
                "desc": "Real-time sign language interpretation using deep learning.",
                "tech": "Computer Vision, Deep Learning"
            },
            {
                "name": "Electronics Design Experiments",
                "desc": "Analog, Digital and VLSI circuit implementations.",
                "tech": "VLSI, Electronics"
            }
        ],

        "skills": [
            "Python",
            "Natural Language Processing",
            "Deep Learning",
            "Electronics & VLSI",
            "GitHub",
            "PyCharm"
        ]
    }

    return templates.TemplateResponse("index.html", context)
