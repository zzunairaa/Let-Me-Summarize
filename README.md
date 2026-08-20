# Let Me Summarize

An AI-powered web application for generating concise abstractive summaries from long-form text. The application combines a FastAPI backend with Hugging Face Transformers and Google's T5 model, providing users with control over summary length and basic text-reduction statistics.

## Overview

Let Me Summarize provides a lightweight interface for text summarization. Users can submit text, select the desired level of compression, and receive an automatically generated summary.

The application uses the `t5-small` model through the Hugging Face Transformers library for abstractive summarization.

### Key Features

- Abstractive text summarization using `t5-small`
- Adjustable summary length from 10% to 70%
- Original and summarized word counts
- Automatic reduction percentage calculation
- Copy-to-clipboard functionality
- REST API built with FastAPI
- Interactive API documentation
- Responsive frontend using HTML, CSS, and JavaScript
- Docker support
- Hugging Face Spaces compatible deployment

---

## Architecture

The application follows a simple client-server architecture:

```text
User Input
    |
    v
Frontend
HTML / CSS / JavaScript
    |
    | POST /summarize
    v
FastAPI Backend
    |
    v
Hugging Face Transformers
    |
    v
T5-small
    |
    v
Generated Summary
    |
    v
Summary Statistics
    |
    v
Frontend Response
```

The frontend sends the source text and requested summary ratio to the FastAPI backend.

Example request:

```json
{
  "text": "Text that should be summarized...",
  "ratio": 0.3
}
```

The backend processes the request using the T5 summarization pipeline and returns the generated summary along with text statistics.

Example response:

```json
{
  "summary": "Generated summary...",
  "original_words": 250,
  "summary_words": 72,
  "reduction_percent": 71.2
}
```

---

## Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning

- Hugging Face Transformers
- T5-small
- PyTorch

### Frontend

- HTML5
- CSS3
- JavaScript

### Deployment

- Docker
- Hugging Face Spaces

---

## Project Structure

```text
Let-Me-Summarize/
|
├── main.py
├── index.html
├── requirements.txt
├── dockerfile
└── README.md
```

### `main.py`

Contains the FastAPI application, request and response models, summarization pipeline, CORS configuration, and API endpoints.

### `index.html`

Contains the frontend implementation, including:

- Text input interface
- Summary-length control
- API integration
- Summary rendering
- Word-count statistics
- Copy functionality
- Interface styling and animations

### `requirements.txt`

Contains the Python dependencies required to run the backend.

### `dockerfile`

Defines the container configuration used to deploy the application.

---

## Getting Started

### Prerequisites

Ensure the following are installed:

- Python 3.9 or later
- `pip`
- Git

Docker is optional if you prefer to run the application in a container.

### Clone the Repository

```bash
git clone <your-repository-url>
cd Let-Me-Summarize
```

### Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

On the first run, Hugging Face Transformers will download the `t5-small` model and cache it locally.

---

## Running the Backend

Start the FastAPI development server:

```bash
uvicorn main:app --reload --port 7860
```

The backend will be available at:

```text
http://localhost:7860
```

---

## API Documentation

FastAPI automatically exposes interactive API documentation.

### Swagger UI

```text
http://localhost:7860/docs
```

### ReDoc

```text
http://localhost:7860/redoc
```

---

## API Reference

### Health Check

```http
GET /
```

Example response:

```json
{
  "status": "Let Me Summarize is running!"
}
```

### Generate Summary

```http
POST /summarize
```

#### Request Body

```json
{
  "text": "Artificial intelligence is transforming software development by allowing machines to perform tasks that previously required human intelligence...",
  "ratio": 0.3
}
```

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `text` | `string` | Source text to summarize. Minimum length: 50 characters. |
| `ratio` | `float` | Target summary ratio between `0.1` and `0.7`. Default: `0.3`. |

#### Example Response

```json
{
  "summary": "Artificial intelligence is changing software development by automating tasks traditionally requiring human intelligence.",
  "original_words": 120,
  "summary_words": 35,
  "reduction_percent": 70.8
}
```

---

## Running the Frontend

The frontend is implemented as a standalone HTML application and can be served using any static HTTP server.

For example:

```bash
python -m http.server 5500
```

Then navigate to:

```text
http://localhost:5500
```

To connect the frontend to a locally running backend, configure the API endpoint in `index.html`:

```javascript
fetch("http://localhost:7860/summarize", {
```

The frontend can also be configured to communicate with a remotely deployed Hugging Face Spaces backend.

---

## Running with Docker

Build the Docker image:

```bash
docker build -t let-me-summarize .
```

Run the container:

```bash
docker run -p 7860:7860 let-me-summarize
```

The API will then be available at:

```text
http://localhost:7860
```

---

## Summarization Model

The application uses Google's **T5-small** model.

T5, or Text-to-Text Transfer Transformer, represents natural language processing tasks using a unified text-to-text architecture.

The summarization pipeline is initialized using Hugging Face Transformers:

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="t5-small"
)
```

The project uses abstractive summarization rather than extractive summarization. This means the model generates new sentences representing the important information from the source instead of simply selecting existing sentences.

`t5-small` provides a relatively lightweight model suitable for demonstrations and applications where infrastructure requirements should remain limited.

---

## Summary Length Control

Users can configure the desired summary ratio between 10% and 70%.

For example, for an input containing 500 words and a selected ratio of 30%, the application uses the ratio to determine an approximate target summary length:

```text
500 × 0.30 = 150
```

The model is therefore instructed to generate a substantially shorter representation of the original input.

The resulting summary may not contain exactly 150 words because transformer models generate sequences using tokens rather than enforcing exact word counts.

Example:

```text
Original text:     500 words
Generated summary: 128 words
Reduction:         74.4%
```

---

## Limitations

The current implementation is intended as a lightweight summarization application and has several limitations:

- `t5-small` has a limited context window for long documents.
- The summary ratio represents an approximate target rather than an exact output length.
- Very large documents cannot be processed directly without additional chunking logic.
- Generated summaries may occasionally omit relevant information.
- The API currently does not implement authentication.
- Rate limiting is not implemented.
- The model is loaded into memory when the application starts.
- CORS configuration should be restricted before production deployment.

---

## Future Improvements

Potential extensions include:

- PDF and document upload support
- Web article and URL summarization
- Long-document chunking
- Hierarchical summarization
- Support for larger summarization models
- Multiple summarization modes
- Bullet-point summary generation
- Key information extraction
- Multilingual summarization
- Streaming responses
- GPU inference
- Model caching and optimization
- Request rate limiting
- API authentication
- Persistent summary history
- Automated testing
- CI/CD workflows
- Improved monitoring and error handling

For long-document support, the architecture could be extended to use hierarchical summarization:

```text
Document
    |
    v
Text Extraction
    |
    v
Chunking
    |
    v
Individual Chunk Summaries
    |
    v
Summary Aggregation
    |
    v
Final Summary
```

This approach would allow documents larger than the model's context window to be processed incrementally.

---

## Deployment

The backend can be deployed as a Docker container or hosted using Hugging Face Spaces.

For a production deployment, additional considerations should include:

- Restricted CORS policies
- Request validation
- Rate limiting
- Authentication
- Logging and monitoring
- Model inference optimization
- Health and readiness checks

---

## License

No open-source license is currently specified.

If this repository is intended for public distribution or external contributions, consider adding an appropriate license such as the MIT License.

---

## Acknowledgements

This project is built using:

- FastAPI
- Hugging Face Transformers
- PyTorch
- Google T5
