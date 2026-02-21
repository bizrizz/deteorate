# Stage 0: clone repo and pull Git LFS so we get real clif_data (Railway build context has only LFS pointers)
ARG REPO_URL=https://github.com/bizrizz/deteorate.git
FROM alpine:3.19 AS lfs
RUN apk add --no-cache git git-lfs && git lfs install
WORKDIR /repo
RUN git clone --depth 1 "${REPO_URL}" . && git lfs pull

# Stage 1: app image
FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app, artifacts, and CLIF data (from LFS stage)
COPY api.py data_extraction.py feature_engineering.py train_model.py labeling.py labeling_sofa2.py ./
COPY output_full_run/ ./output_full_run/
COPY configs/ ./configs/
COPY --from=lfs /repo/clif_data/ ./clif_data/

ENV DETECTORATE_ARTIFACTS_DIR=/app/output_full_run
ENV CLIF_DATA_DIR=/app/clif_data

EXPOSE 8000
ENV PORT=8000
# Start via Python so PORT is read from env (Railway may not expand $PORT in Procfile)
CMD ["python", "api.py"]
