
#http://localhost:8000/docs

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
import uvicorn

app = FastAPI(
    title = "Let Me Summarize API",
    description = "Paste any text, get a clean summary using T5",
    version = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading T5 model, please wait...")
summarizer = pipeline("summarization", model="t5-small")
print(" Model loaded and ready!")


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, description="The text to summarize")
    ratio: float = Field(0.3, ge=0.1, le=0.7, description="Summary ratio 0.1 to 0.7")

class SummarizeResponse(BaseModel):
    summary: str
    original_words: int
    summary_words: int
    reduction_percent: float


@app.get("/")
def health_check():
    return {"status": "Let Me Summarize is running!"}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):

    # Step 1 — count original words
    original_words = len(req.text.split())

    # Step 2 — calculate min and max length from ratio
    max_length = max(50, int(original_words * req.ratio))
    min_length = max(20, int(max_length * 0.5))

    # Step 3 — run the T5 model
    result = summarizer(
        req.text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )

    # Step 4 — extract the summary text
    summary_text = result[0]["summary_text"]

    # Step 5 — count summary words
    summary_words = len(summary_text.split())

    # Step 6 — calculate reduction percentage
    reduction = round(100 - (summary_words / original_words * 100), 1)

    # Step 7 — return the response
    return SummarizeResponse(
        summary=summary_text,
        original_words=original_words,
        summary_words=summary_words,
        reduction_percent=reduction
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)