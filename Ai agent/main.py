from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pdfplumber
import requests
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")

MISTRAL_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MISTRAL_API_KEY = "nvapi-JaT_0U16j9En1f9DPhgaTiRL9KIZpTBAYeVTvNtN6h7M5EcA" #Add your own API key here
#API key can be generated on https://build.nvidia.com/nv-mistralai/mistral-nemo-12b-instruct
#I have personally chosen mistral you can use any other as per your choice
pdf_content = None

history = []

def get_mistral_response(question, context):
    """
    Call Mistral's API to generate an answer based on the question and context.
    """
    prompt = f"Answer the question based on the following context: \n{context}\n\nQuestion: {question}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    data = {
        "model": "nv-mistralai/mistral-nemo-12b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 200
    }
    response = requests.post(MISTRAL_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        if "choices" in response_json:
            answer = response_json["choices"][0]["message"]["content"].strip()
            return answer.replace("**", "<strong>").replace("**", "</strong>")
        else:
            return "The AI model did not return a valid answer."
    else:
        return "Failed to get a response from the AI model."

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def get_answer(request: Request, pdf_file: UploadFile = File(None), question: str = Form(...)):
    global pdf_content, history

    
    if pdf_file:
        file_location = f"temp_{pdf_file.filename}"
        with open(file_location, "wb") as file:
            file.write(await pdf_file.read())

        extracted_text = ""
        try:
            with pdfplumber.open(file_location) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text
        except Exception as e:
            return JSONResponse({"answer": "Failed to extract text from the PDF."}, status_code=400)
        finally:
            os.remove(file_location)

        if extracted_text.strip():
            pdf_content = extracted_text
        else:
            return JSONResponse({"answer": "Uploaded PDF is empty or couldn't be read."}, status_code=400)

    if not pdf_content:
        return JSONResponse({"answer": "No PDF uploaded or extracted text is empty."}, status_code=400)

    answer = get_mistral_response(question, pdf_content)

    history.append({"question": question, "answer": answer})

    return JSONResponse({"answer": answer})

@app.get("/history", response_class=HTMLResponse)
async def get_history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request, "history": history})

@app.post("/clear_history", response_class=HTMLResponse)
async def clear_history(request: Request):
    global history
    history = []  
    return RedirectResponse(url="/history", status_code=303)
