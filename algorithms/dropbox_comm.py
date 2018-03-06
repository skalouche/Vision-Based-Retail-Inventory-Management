import dropbox
import argparse
import contextlib
import datetime
import os
import six
import sys
import time
import unicodedata
from dropbox.files import FileMetadata, FolderMetadata
from termcolor import colored   # for printing to terminal in colored text
import ipdb


############# Functions ###############
def download(dbx, folder, subfolder, name):
    """Download a file.
    Return the bytes of the file, or None if it doesn't exist.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    with stopwatch('download'):
        try:
            md, res = dbx.files_download(path)
        except dropbox.exceptions.HttpError as err:
            print('*** HTTP error', err)
            return None
    data = res.content
    print(len(data), 'bytes; md:', md)
    return data

def upload(dbx, fullname, folder, subfolder, name, overwrite=False):
    """Upload a file.
    Return the request response, or None in case of error.

    dbx = handle to dropbox object
    fullname = full path of file to be uploaded
    folder = higher tier folder to save file to on dropbox
    subfolder = lower tier folder to save file to on dropbox
    name = file name when file is loaded to dropbox
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    mode = (dropbox.files.WriteMode.overwrite
            if overwrite
            else dropbox.files.WriteMode.add)
    mtime = os.path.getmtime(fullname)
    with open(fullname, 'rb') as f:
        data = f.read()
    with stopwatch('upload %d bytes' % len(data)):
        try:
            res = dbx.files_upload(
                data, path, mode,
                client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                mute=True)
        except dropbox.exceptions.ApiError as err:
            print('*** API error', err)
            return None
    print colored('uploaded as:', 'green'), colored(res.name.encode('utf8'), 'green')
    return res

@contextlib.contextmanager
def stopwatch(message):
    """Context manager to print how long a block of code took."""
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        # print('Total elapsed time for %s: %.3f' % (message, t1 - t0))


# ############# MAIN ################
# # get access token to authenticate session
# dbx = dropbox.Dropbox('<dropbox_auth_id>')

# # upload file '1.JPG' to main directory (main directory is indicated by parameters folder='', and subfolder='') with a name 'first_pic.JPG'
# # NOTE: an error occurs if you try to upload a file to the directory if there is already a file with the same name in that directory
# # upload(dbx,'1.JPG','','','first_pic.JPG')

# ipdb.set_trace()

# # download file ''
# download(dbx,'','','2016_04_23__15_55_40.jpg')


