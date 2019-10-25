import requests
from config.Json_read import JsonObj

class downloadData:

    def __init__(self):
        obj=JsonObj()
        self.file_url=obj.get_config_obj('file_url_download')
        self.file_name=obj.get_config_obj('file_name')
        self.data_store_location=obj.get_config_obj('data_store_location')

    def s3_download_data(self):
        r = requests.get(self.file_url, stream=True)
        with open(self.data_store_location+self.file_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                # writing one chunk at a time to csv file
                if chunk:
                    f.write(chunk)