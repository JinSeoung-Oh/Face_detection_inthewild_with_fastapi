# Copyright 2022 The vedadet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
 
