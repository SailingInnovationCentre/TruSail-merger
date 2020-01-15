import azure.functions as func

import logging
import zipfile
import tempfile
import os

from parallel import *
from main import main

def main(myblob: func.InputStream, outputblob: func.Out[func.InputStream]):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")

    with tempfile.TemporaryDirectory() as tmpdirname:
        #outputblob.set('created temporary directory ' + tmpdirname)
        zipfilename = tmpdirname + "/input.zip"
        with open(zipfilename, 'w+b') as z:
            z.write(myblob.read())
        zf = zipfile.ZipFile(zipfilename)
        extractdir = tmpdirname + "/files"
        zf.extractall(extractdir)

        main(extractdir)

        #list_files = os.listdir(extractdir)
        #outputblob.set("Aantal files in ZIP file: " + str(len(list_files)))

