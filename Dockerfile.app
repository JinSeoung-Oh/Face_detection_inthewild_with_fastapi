FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install google google-cloud google-cloud-vision
RUN pip install --upgrade google-cloud-storage google_auth_httplib2 google-api-python-client

COPY ./requirements.txt /home/app/
WORKDIR /home/app/
RUN pip install -r requirements.txt

COPY ./ /home/app
WORKDIR /home/app
CMD python3 gateway.py
