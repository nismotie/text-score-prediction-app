import imaplib as iml
from dotenv import load_dotenv
import os
load_dotenv()

class GetMail(object):

    """Docstring for GetMail. """

    def __init__(self):
        self.mail_user = os.getenv("EMAIL_USERNAME")
        self.mail_pass = os.getenv("EMAIL_USERNAME")

        self.mail = iml.IMAP4_SSL("imap.google.com", port=993)

        
