import json
import os
import sys
import base64
import gridfs
import pymongo
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['sec_forms']

#fs = gridfs.GridFS(db)
collection = db['10Q_text']
collection.create_index([("cik",pymongo.ASCENDING)])
print("total ", len(collection.distinct('cik')), " CIK in 10Q")

collection = db['10K_text']
collection.create_index([("cik",pymongo.ASCENDING)])
print("total ", len(collection.distinct('cik')), " CIK in 10K")
#col = collection.distinct('cik')
