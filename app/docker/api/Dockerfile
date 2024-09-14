FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    curl \
    libgomp1

RUN pip3 install -r requirements.txt

EXPOSE 80

ENTRYPOINT ["python", "api.py"]