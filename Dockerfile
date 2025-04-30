FROM python:3.10

RUN pip install rcsb-embedding-model

RUN useradd --create-home embedding-user
USER embedding-user
WORKDIR /home/embedding-user

COPY . /home/embedding-user
RUN pip install --no-cache-dir .

ENTRYPOINT ["/bin/sh"]
