FROM python:3.10

WORKDIR /app
COPY . /app/

RUN pip install -e .

ENTRYPOINT ["/bin/sh"]
