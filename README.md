# GeoSpaider

GeoSpaider is a lightweight geospatial data pipeline and RAG (Retrieval-Augmented Generation) system for querying Earth Observation (EO) imagery and metadata.

---

## 🚀 Features
- Dynamic querying of satellite imagery (metadata-first, with previews).
- Supports filtering and retrieval from metadata CSVs.
- Image previews for quick inspection.
- Minimalistic and clean Python structure.
- Containerized deployment with Docker (optional).

---

## 🛠️ Tech Stack

I chose the following stack to balance performance, simplicity, and open-source availability:

- **PyTorch 2.8.0** → `torch==2.8.0`  
  Deep learning backbone, used for embeddings / model inference.

- **Transformers 4.55.2** → `transformers==4.55.2`  
  Language model utilities for query understanding.

- **ChromaDB 1.0.20** → `chromadb==1.0.20`  
  Lightweight vector database for similarity search.

- **LangChain 0.3.27 (+ integrations)** →  
  `langchain==0.3.27`  
  `langchain-chroma==0.2.5`  
  `langchain-community==0.3.27`  
  Framework for RAG orchestration.

- **Data utilities** → `pandas==2.3.1`, `numpy==2.2.2`

- **Image handling** → `pillow==11.3.0`, `matplotlib==3.10.5`, `scikit-image==0.25.2`, `imageio==2.37.0`

- **Environment management** → `python-dotenv==1.1.1`

- **Optional LLM integration** → `google-generativeai==0.8.5` (Gemini API)

- **Docker/Podman** → Reproducible, portable containerized runs.



## 📂 Project Structure

GeoSpaider/
├── bhopal_s2/              # Example Sentinel-2 imagery + metadata
├── notebook/               # Jupyter notebooks (experiments)
├── src/                    # Core source code
│   ├── demo.py             # Demo entrypoint
│   ├── embed.py            # Embedding + vector store builder
│   ├── fetch.py            # Fetch utilities for imagery/metadata
│   ├── llm.py              # LLM integration (transformers, Google GenAI)
│   └── rag.py              # RAG pipeline (retrieval + generation)
├── vector_store/           # Chroma vector DB
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
└── README.md               # Documentation


## 🎯 Example Queries

Run inside container or locally:

python -m src.demo --open-images

Example prompts:

show the most recent image of Bhopal

show images with lowest cloud cover

give me the highest cloud cover image available

show me two random images for diversity

Expected output:

Top 1: S2A_43QGF_20240828_0_L2A | date=2024-08-28T05:32:33Z | cloud=27.7
[preview] bhopal_s2/previews/S2A_43QGF_20240828_0_L2A.jpg


## ⚡ Setup

git clone https://github.com/amby005/GeoSpaider.git
cd GeoSpaider

python -m venv .venv
# Activate:
# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt

# Run demo
python -m src.demo --open-images



Docker (Optional)
Build the image:
docker build -t geospaider .


Run with mounted datasets:

On Windows (PowerShell):

docker run --rm -it `
  -v C:\Users\amber\Downloads\GeoSpaider\bhopal_s2:/app/bhopal_s2 `
  -v C:\Users\amber\Downloads\GeoSpaider\vector_store:/data/vector_store `
  geospaider python -m src.demo --open-images


📌 Notes & Improvements
Notes & Improvements

Current Docker image is ~12GB (due to PyTorch + Transformers). Can be reduced by using python:3.11-slim and smaller model weights.

Could add a docker-compose.yaml for easier runs.

Additional EO datasets and better cloud-filtering can be integrated.

Future: Automatic download of new Sentinel-2 data.
