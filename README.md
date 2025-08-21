GeoSpaider

GeoSpaider is a lightweight geospatial data pipeline and RAG (Retrieval-Augmented Generation) system for querying Earth Observation (EO) imagery and metadata.

🚀 Features

Dynamic querying of satellite imagery (metadata-first, with previews).

Supports filtering and retrieval from metadata CSVs.

Image previews for quick inspection.

Minimalistic and clean Python structure.

Containerized deployment with Docker (optional).

🛠️ Tech Stack

I chose the following stack to balance performance, simplicity, and open-source availability:

PyTorch 2.8.0 → torch==2.8.0
Deep learning backbone, used for embeddings / model inference.

Transformers 4.55.2 → transformers==4.55.2
Language model utilities for query understanding.

ChromaDB 1.0.20 → chromadb==1.0.20
Lightweight vector database for similarity search.

LangChain 0.3.27 (+ integrations) →

langchain==0.3.27

langchain-chroma==0.2.5

langchain-community==0.3.27
Framework for building the RAG pipeline and connecting with Chroma.

Pandas 2.3.1 / NumPy 2.2.2 →

pandas==2.3.1

numpy==2.2.2
Data wrangling & numerical utilities.

Image Handling & Visualization →

pillow==11.3.0

matplotlib==3.10.5

scikit-image==0.25.2

imageio==2.37.0
For loading, saving, and previewing satellite images.

python-dotenv 1.1.1 → python-dotenv==1.1.1
For managing API keys / environment variables.

Google Generative AI 0.8.5 → google-generativeai==0.8.5
Optional Gemini API support for text/image description.

Docker/Podman → Reproducible, portable containerized runs.

📂 Project Structure
GeoSpaider/
│── bhopal_s2/              # Example Sentinel-2 imagery + metadata
│── notebook/               # Jupyter notebooks (experiments)
│── src/                    # Core source code
│   ├── demo.py             # Demo entrypoint
│   ├── embed.py            # Embedding + vector store builder
│   ├── fetch.py            # Fetch utilities for imagery/metadata
│   ├── llm.py              # LLM integration (transformers, Google GenAI)
│   └── rag.py              # RAG pipeline (retrieval + generation)
│── vector_store/           # Chroma vector DB
│── requirements.txt        # Python dependencies
│── Dockerfile              # Container definition
│── README.md               # Documentation

⚡ Setup
Local (Python)
git clone https://github.com/<your-username>/GeoSpaider.git
cd GeoSpaider
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m src.demo --open-images

Docker (Optional, reproducible)

Build the image:

docker build -t geospaider .


Run with mounted datasets:

docker run --rm -it ^
  -v C:\Users\amber\Downloads\GeoSpaider\bhopal_s2:/app/bhopal_s2 ^
  -v C:\Users\amber\Downloads\GeoSpaider\vector_store:/data/vector_store ^
  geospaider python -m src.demo --open-images