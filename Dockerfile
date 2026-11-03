# Urban Heat Hotspots dashboard (live mode: satellite data is fetched on demand, so the container needs internet access).
#   docker build -t urban-heat .
#   docker run -p 8501:8501 urban-heat          # then open http://localhost:8501
# Needs Deploy/best.pt and Deploy/meta.json in the build context (see Deploy/README.md). Plan for 4 GB of RAM or more.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app
COPY requirements-app.txt .
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-app.txt

COPY dashboard dashboard
COPY extraction extraction
COPY pre_processing pre_processing
COPY models models
COPY .streamlit .streamlit
COPY Deploy Deploy

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
