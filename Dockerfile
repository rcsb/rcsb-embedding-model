FROM python:3.10

WORKDIR /app
COPY . /app/

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["inference"]
