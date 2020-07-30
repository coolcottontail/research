from pymongo import MongoClient
import base64
import gridfs
import os
import bs4 as bs
import unicodedata
from tqdm import tqdm
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

counter = Counter(0)

def RemoveTags(soup):
    '''
    Drops HTML tags, newlines and unicode text from
    filing text.

    Parameters
    ----------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup.

    Returns
    -------
    text : str
        Filing text.

    '''

    # Remove HTML tags with get_text
    text = soup.get_text()

    # Remove newline characters
    text = text.replace('\n', ' ')

    # Replace unicode characters with their
    # "normal" representations
    text = unicodedata.normalize('NFKD', text)

    return text

def RemoveNumericalTables(soup):
    '''
    Removes tables with >15% numerical characters.

    Parameters
    ----------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup.

    Returns
    -------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup
        with numerical tables removed.

    '''

    # Determines percentage of numerical characters
    # in a table
    def GetDigitPercentage(tablestring):
        if len(tablestring) > 0.0:
            numbers = sum([char.isdigit() for char in tablestring])
            length = len(tablestring)
            return numbers / length
        else:
            return 1

    # Evaluates numerical character % for each table
    # and removes the table if the percentage is > 15%
    [x.extract() for x in soup.find_all('table') if GetDigitPercentage(x.get_text()) > 0.15]

    return soup

def ConvertHTML_10K(cik):
    client = MongoClient('localhost', 27017)
    db = client['sec_forms']
    fs = gridfs.GridFS(db)

    collection = db['10K']
    textCollection = db['10K_text']

    documents = collection.find({"cik":cik},{"_id":0})
    counter.increment()

    row = {}
    row['cik'] = cik

    # Iterate over scraped files and clean
    for doc in tqdm(documents):
        str = doc['name']
        row['date'] = str[str.find('_')+1:str.find('.')]
        if doc['gridfs'] == 0:
            data = base64.b64decode(doc['file']).decode("utf-8", "ignore")
        else:

            data = base64.b64decode(fs.get(doc['gridfs']).read()).decode("utf-8", "ignore")
        soup = bs.BeautifulSoup(data, "lxml")
        soup = RemoveNumericalTables(soup)
        row['text'] = RemoveTags(soup)
        #textCollection.update_one(row, {"$set": row}, upsert=True)
        textCollection.insert_one(row)
        del row["_id"]
    return


def ConvertHTML_10Q(cik):
    client = MongoClient('localhost', 27017)
    db = client['sec_forms']
    fs = gridfs.GridFS(db)

    collection = db['10Q']
    textCollection = db['10Q_text']

    documents = collection.find({"cik":cik},{"_id":0})
    counter.increment()

    row = {}
    row['cik'] = cik

    # Iterate over scraped files and clean
    for doc in tqdm(documents):
        str = doc['name']
        row['date'] = str[str.find('_')+1:str.find('.')]
        if doc['gridfs'] == 0:
            data = base64.b64decode(doc['file']).decode("utf-8", "ignore")
        else:

            data = base64.b64decode(fs.get(doc['gridfs']).read()).decode("utf-8", "ignore")
        soup = bs.BeautifulSoup(data, "lxml")
        soup = RemoveNumericalTables(soup)
        row['text'] = RemoveTags(soup)
        #textCollection.update_one(row, {"$set": row}, upsert=True)
        textCollection.insert_one(row)
        del row["_id"]
    return



def ComputeCosineSimilarity(words_A, words_B):
    '''
    Compute cosine similarity between document A and
    document B.

    Parameters
    ----------
    words_A : set
        Words in document A.
    words_B : set
        Words in document B

    Returns
    -------
    cosine_score : float
        Cosine similarity between document
        A and document B.

    '''

    # Compile complete set of words in A or B
    words = list(words_A.union(words_B))

    # Determine which words are in A
    vector_A = [1 if x in words_A else 0 for x in words]

    # Determine which words are in B
    vector_B = [1 if x in words_B else 0 for x in words]

    # Compute cosine score using scikit-learn
    array_A = np.array(vector_A).reshape(1, -1)
    array_B = np.array(vector_B).reshape(1, -1)
    cosine_score = cosine_similarity(array_A, array_B)[0, 0]

    return cosine_score


def ComputeJaccardSimilarity(words_A, words_B):
    '''
    Compute Jaccard similarity between document A and
    document B.

    Parameters
    ----------
    words_A : set
        Words in document A.
    words_B : set
        Words in document B

    Returns
    -------
    jaccard_score : float
        Jaccard similarity between document
        A and document B.

    '''

    # Count number of words in both A and B
    words_intersect = len(words_A.intersection(words_B))

    # Count number of words in A or B
    words_union = len(words_A.union(words_B))

    # Compute Jaccard similarity score
    jaccard_score = words_intersect / words_union

    return jaccard_score

def ComputeSimilarityScores10K(cik):
    '''
    Computes cosine and Jaccard similarity scores
    over 10-Ks for a particular CIK.

    Parameters
    ----------
    cik: str
        Central Index Key used to scrape and name
        files.

    Returns
    -------
    None.

    '''

    # Open the directory that holds plaintext
    # filings for the CIK
    os.chdir(cik + '/rawtext')
    print("Parsing CIK %s..." % cik)

    # Get list of files to over which to compute scores
    # excluding hidden files and directories
    file_list = [fname for fname in os.listdir() if not
    (fname.startswith('.') | os.path.isdir(fname))]
    file_list.sort()

    # Check if scores have already been calculated...
    try:
        os.mkdir('../metrics')
    # ... if they have been, exit
    except OSError:
        print("Already parsed CIK %s..." % cik)
        os.chdir('../..')
        return

    # Check if enough files exist to compute sim scores...
    # If not, exit
    if len(file_list) < 2:
        print("No files to compare for CIK", cik)
        os.chdir('../..')
        return

    # Initialize dataframe to store sim scores
    dates = [x[-14:-4] for x in file_list]
    cosine_score = [0] * len(dates)
    jaccard_score = [0] * len(dates)
    data = pd.DataFrame(columns={'cosine_score': cosine_score,
                                 'jaccard_score': jaccard_score},
                        index=dates)

    # Open first file
    file_name_A = file_list[0]
    with open(file_name_A, 'r') as file:
        file_text_A = file.read()

    # Iterate over each 10-K file...
    for i in range(1, len(file_list)):
        file_name_B = file_list[i]

        # Get file text B
        with open(file_name_B, 'r') as file:
            file_text_B = file.read()

        # Get set of words in A, B
        words_A = set(re.findall(r"[\w']+", file_text_A))
        words_B = set(re.findall(r"[\w']+", file_text_B))

        # Calculate similarity scores
        cosine_score = ComputeCosineSimilarity(words_A, words_B)
        jaccard_score = ComputeJaccardSimilarity(words_A, words_B)

        # Store score values
        date_B = file_name_B[-14:-4]
        data.at[date_B, 'cosine_score'] = cosine_score
        data.at[date_B, 'jaccard_score'] = jaccard_score

        # Reset value for next loop
        # (We don't open the file again, for efficiency)
        file_text_A = file_text_B

    # Save scores
    os.chdir('../metrics')
    data.to_csv(cik + '_sim_scores.csv', index=False)
    os.chdir('../..')


client = MongoClient('localhost', 27017)
db = client['sec_forms']
fs = gridfs.GridFS(db)


"""
collection = db['10K']
textCollection = db['10K_text']

ciks = list(collection.distinct('cik'))
print("CIK length is ", len(ciks))
print(type(ciks))
# Setup a list of processes that we want to run
pool = Pool(processes=16)
pool.map(ConvertHTML_10K, ciks)
pool.close()

print("Total ", counter.value(), " CIK are converted")
"""

collection = db['10Q']
textCollection = db['10Q_text']

ciks = list(collection.distinct('cik'))
print("CIK length is ", len(ciks))
print(type(ciks))
# Setup a list of processes that we want to run
pool = Pool(processes=16)
pool.map(ConvertHTML_10Q, ciks)
pool.close()

print("Total ", counter.value(), " CIK are converted")