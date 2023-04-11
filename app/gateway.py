from requests.api import head
from fastapi import FastAPI, Header, requests
from pydantic.main import BaseModel
from typing import Optional
import os

import uvicorn
#from gunicorn.app.wsgiapp import WSGIApplication
import json
import time
import os
#import logging
import requests
import numpy as np
import shutil
from celery import Celery

from google.cloud import storage
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from tools.infer import infer_face

class FromFrontendRequest(BaseModel):
    model_mode: str
    gcs_file_path: str

class gunicornApp(WSGIApplication):
    def __init__(self, app_uri, options=None):
        self.options = options or {}
        self.app_uri = app_uri
        super().__init__()
        
    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
            }
        for key, value in config.items():
            self.cfg.set(key.lower(), value) 
    
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def gunicorn_run():
    options = {
       "bind": "0.0.0.0:8180",
       "workers": 2,
       "gthreads" : 4,
       "worker-connections": 300,
       "timeout" : 1500,
       "worker_class": "uvicorn.workers.UvicornWorker",
       "preload":True,
       "reload": True
       #"ssl_keyfile" : 
       #"ssl_certfile": 
       }
    gunicornApp("gateway:app", options).run()
       

def make_dataset_folder(directory):
    
    items = os.listdir(directory)
    items = [(os.path.join(directory, f)) for f in items]
    items = sorted(items)

    #print(f'Found {len(items)} folder imgs')

    return items 

CELERY_BROKER_URL = 'redis://redis:6379'
CELERY_RESULT_BACKEND = 'redis:redis:6379'

papp = Celery('tasks', broker=CELERY_BROKER_URL, backend = CELERY_RESULT_BACKEND)

@app.get("/")
async def root():
    return {"message" : "face detection API page"}

@app.post(path='/tinaNet')
async def sfa3d_vox(request: FromFrontendRequest,referer: Optional[str] = Header(None, convert_underscores=False)):
          
    start_time = time.time()
    model_mode = request.model_mode
    gcs_file_path = request.file_path
    #print(torch.multiprocessing.get_start_method())
    
    os.makedirs('./static/'+name, exist_ok=True)
    #print(torch.cuda.is_available())
    storage_client = storage.Client.from_service_account_json('your service account json file path')
    bucket = stroage_client.get_bucket('your bucket name')
    gcs_link = 'yout gcs_link'
    blobs = bucket.blob(gcs_link)
      
    blobs.download_to_filename('your_file_path')
    
    task = papp.send_task(name='tasks.face',
                          args = ['your_file_path'],
                          kwargs={})    
    return task.id
    
if __name__ == '__main__':         
    uvicorn.run("gateway:app",
                host="0.0.0.0",
                port=8280,
                reload=True,
                workers=1,
                )

