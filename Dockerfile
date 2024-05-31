FROM python:3.9-alpine

WORKDIR /code

# Install necessary packages and glibc
RUN apk --no-cache add ca-certificates wget \
    && wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub \
    && wget https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.33-r0/glibc-2.33-r0.apk \
    && apk add glibc-2.33-r0.apk \
    && apk add --no-cache bash gcompat libstdc++ \
    && rm -rf /var/cache/apk/*

COPY database/pipeline.pkl /code/database/pipeline.pkl
COPY requirements.txt /code/requirements.txt
COPY setup.py /code/setup.py
COPY knowledge_database /code/knowledge_database
COPY api /code/api

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

RUN --mount=type=secret,id=OPENAI_API_KEY sh -c 'echo "export OPENAI_API_KEY=$(cat /run/secrets/OPENAI_API_KEY)" >> /etc/profile.d/openai.sh'

CMD ["/bin/bash", "-c", "source /etc/profile && uvicorn api.api:app --host 0.0.0.0 --port 8080"]
