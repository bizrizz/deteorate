# Faster Railway builds: cache pip install, then copy app + artifacts
FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy app and artifacts
COPY api.py data_extraction.py feature_engineering.py train_model.py labeling.py labeling_sofa2.py ./
COPY output_full_run/ ./output_full_run/
COPY configs/ ./configs/

ENV DETECTORATE_ARTIFACTS_DIR=/app/output_full_run

EXPOSE 8000
ENV PORT=8000
CMD ["/bin/sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
