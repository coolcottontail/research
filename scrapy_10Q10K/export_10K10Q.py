import json
from pymongo import MongoClient
import sys
import base64
import gridfs

client = MongoClient('localhost', 27017)
db = client['sec_forms']
collection = db['10Q']
fs = gridfs.GridFS(db)

rows = list(collection.find({},{"_id":0}))

for row in rows:
    f = open(row['name'], "w")
    if (row['gridfs'] == 0):
        decoded_html = base64.b64decode(row['file']).decode("utf-8", "ignore")
    else:
        decoded_html = base64.b64decode(fs.get(row['gridfs']).read()).decode("utf-8", "ignore")
    print(row['name'],  " size is ", sys.getsizeof(decoded_html))
    f.write(decoded_html)
    f.close()