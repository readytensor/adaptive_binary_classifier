FROM python:3.8.0-slim as builder


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         ca-certificates \
         dos2unix \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements/requirements.txt /opt/
RUN pip3 install -r /opt/requirements.txt 

COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/fix_line_endings.sh 

COPY src ./opt/src

RUN /opt/fix_line_endings.sh

WORKDIR /opt/src


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"

RUN chmod +x train.py \
 && chmod +x predict.py \
 && chmod +x serve.py 

USER 1000
