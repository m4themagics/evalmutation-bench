FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir uv==0.11.16 \
    && uv sync --frozen --extra service --extra llm --no-dev \
    && useradd --create-home --uid 10001 evalmut
COPY data /app/data
USER evalmut
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "evalmutationbench.service.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
