__author__ = 'teemu kanstren'

import dropbox
import os

access_token = os.environ["DROPBOX_TOKEN"]
dbx = dropbox.Dropbox(access_token)
print(dbx.files_list_folder("")) #empty string equals root, which equals the dir the token has access to
filename = "bob"
dbx.files_download_to_file(filename, '/'+ filename)

