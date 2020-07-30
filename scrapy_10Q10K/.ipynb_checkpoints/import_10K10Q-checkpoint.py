import json
import os
import sys
import base64
import gridfs
from tqdm import tqdm
from pymongo import MongoClient
from multiprocessing import Pool
from multiprocessing import Value, Lock

class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


LARGEST_FILE_SIZE = 8000000
counter = Counter(0)

def parse_CIK(cik,collection,fs):
    global counter
    # Look for files scraped for that CIK
    try:
        os.chdir(cik)
    # ...if we didn't scrape any files for that CIK, exit
    except FileNotFoundError:
        print("Could not find directory for CIK", cik)
        return

    print("Parsing CIK %s..." % cik)

    row = {}
    row['cik'] = cik

    # Get list of scraped files
    # excluding hidden files and directories
    file_list = [fname for fname in os.listdir() if not (fname.startswith('.') | os.path.isdir(fname))]
    if (len(file_list) > 0):
        counter.increment()

    # Iterate over scraped files and clean
    for filename in tqdm(file_list):
        row['name'] = filename
        row['gridfs'] = 0
        # print(filename)
        with open(filename, 'r') as file:
            # base64 encoding requires 8 bit bytes as input
            row['file'] = base64.b64encode(bytes(file.read(), "utf-8"))
            # print(filename, " size is ", sys.getsizeof(row['file']))
            if (sys.getsizeof(row['file']) >= LARGEST_FILE_SIZE):
                row['gridfs'] = fs.put(row['file'], filename=filename)
                row['file'] = ''
                # print('file too large, put to grid FS with id ', row['gridfs'])
            # collection.update_one(row, {"$set": row}, upsert=True)
            collection.insert_one(row)
            del row["_id"]

    os.chdir('..')

def import_html_mongo_10Q(cik):
    client = MongoClient('localhost', 27017)
    db = client['sec_forms']
    collection = db['10Q']
    fs = gridfs.GridFS(db)

    os.chdir("/opt/melody/research/10Q10K/10Q")

    parse_CIK(cik,collection,fs)
    return


def import_html_mongo_10K(cik):
    client = MongoClient('localhost', 27017)
    db = client['sec_forms']
    collection = db['10K']
    fs = gridfs.GridFS(db)
    os.chdir("/opt/melody/research/10Q10K/10K")

    parse_CIK(cik,collection,fs)
    return

""" 
os.chdir("/opt/melody/research/10Q10K/10Q")
file_list = [fname for fname in os.listdir() if not fname.startswith('.') and os.path.isdir(fname)]
print("10Q dir size is ", len(file_list))
# Setup a list of processes that we want to run
pool = Pool(processes=16)
pool.map(import_html_mongo_10Q, file_list)
pool.close()

print("Total ", counter.value(), " CIK are imported")
"""


os.chdir("/opt/melody/research/10Q10K/10K")
file_list = [fname for fname in os.listdir() if not fname.startswith('.') and os.path.isdir(fname)]
print("10K dir size is ", len(file_list))

# Setup a list of processes that we want to run
pool = Pool(processes=16)
pool.map(import_html_mongo_10K, file_list)
pool.close()

print("Total ", counter.value(), " CIK are imported")