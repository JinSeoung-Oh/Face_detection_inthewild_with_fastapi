from zipfile import ZipFile
import requests
import os
import glob
import json
from io import BytesIO
import shutil
from pathlib import Path

def extract_zip(gcs_path, zip_path):
    zip_url = requests.get(gcs_path)
    zipfile = ZipFile(BytesIO(zip_url.content))
    zipfile.extractall(zip_path)
    zipfile.close()
    
    
def make_sucess_zip(file_, out_path, request_id):
    zip_path = str(request_id) + '_applied_result.zip'
    zip_file = ZipFile(zip_path, 'w')
    for f in Path(file_).rglob("*"):
        zip_file.write(f, f.name)
    zip_file.close()
    out = out_path + '/'
    
    shutil.move(zip_path, out)
 
