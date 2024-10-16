import os
from google.oauth2 import service_account
from googleapiclient.discovery import build, Resource
from config import SCOPES, CREDENTIALS_PATH

def authenticate_google_drive() -> Resource | None:
    print(f"Current working directory: {os.getcwd()}")

    if not os.path.exists(CREDENTIALS_PATH):
        print(f"Error: {CREDENTIALS_PATH} not found.")
        return None

    try:
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_PATH, scopes=SCOPES)
        
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Error authenticating with service account: {str(e)}")
        return None