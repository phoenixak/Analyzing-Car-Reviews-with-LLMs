# CPU image with the analyzer; use it as a base for batch jobs or wrap Analyzer in your own API.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --uid 1000 --create-home app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install ".[entities]"
USER app
CMD ["python", "-c", "from review_intel.infer import Analyzer; import json, sys; a = Analyzer(); print(json.dumps([vars(x) for x in a.analyze(sys.stdin.read()).aspects], indent=2))"]
