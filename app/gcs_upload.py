from google.cloud import storage
import pandas as pd
import requests
import datetime
from pytz import timezone
import time

storage_client = storage.Client.from_service_account_json('enter your key json')
adapter = requests.adapters.HTTPAdapter(pool_connections=200, pool_maxsize=200, max_retries=5)
storage_client._http.mount("https://", adapter)
storage_client._http._auth_request.session.mount("https://", adapter)

def upload_blob(bucket_name, file_path):
    file_ = file_path.split('/')
    if len(file_) == 3:
       file_path_ = file_[1] + '/' + file_[2]
    else:
         file_path_ = file_[1]
        
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = f"result/{now_time}/{file_path_}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    
    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"

def convert_vaild_gcs_address_to_csv(addresses, file_dir):
    df_name = f"{file_dir}.csv"
    #print(addresses)          
    if len(addresses)<=1:
        flag = None
        info_function=lambda x: x.split("/")[-1].split(".")[0].split('_')
        file_info = [{"original_name": "_".join(info_function(address)[:-1]),
        "download_address":address} for address in addresses]
    else:
         flag = True
         del addresses[zip_index]
         info_function=lambda x: x.split("/")[-1].split(".")[0].split('_')
         file_info = [{"original_name": "_".join(info_function(address)[:-1]),
        "download_address":address} for address in addresses]
         
    df = pd.DataFrame(file_info)
    df.sort_values(['original_name'], ignore_index=True, inplace=True)
    
    if flag == True:
       index = len(df)
       new_row = [zip_address.split('/')[-1].split('.')[-2], zip_address]
       df.loc[index] = new_row
       df.to_csv(df_name)
    elif flag == None:
         df.to_csv(df_name)
    
    return df_name
