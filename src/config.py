import os

# Project structure
OUTPUT_DIR = 'create_RHEM_inputs'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', OUTPUT_DIR)

# Ensure output directory exists
os.makedirs(FULL_OUTPUT_DIR, exist_ok=True)

# Date range for data collection
START_DATE = '2015-09-01'
END_DATE = '2024-10-10'

# Google API configuration
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, 'config', 'rangeveg-0846ba3ac694.json')
TOKEN_PATH = os.path.join(PROJECT_ROOT, 'config', 'token.json')

# Set Google credentials environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH