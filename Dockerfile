FROM python:3.9 AS builder

WORKDIR /gesture-classifier
COPY requirements.txt .

RUN apt-get update

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
