# DowellLogoScan

DowellLogoScan is a web application for logo detection and feature extraction using FastAPI and MongoDB. This project provides functionalities for uploading logo images and searching for similar logos based on extracted features.

## Features

- **User Authentication:** Secure endpoints for uploading and searching logos.
- **Logo Uploading:** Users can upload logo images along with additional metadata.
- **Logo Searching:** Search for similar logos based on features extracted from uploaded images.
- **Responsive Design:** Simple and user-friendly HTML interface for easy navigation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/LL08-MathematicalModelling-dowell/100091-LogoScan.git
   cd logo_scan_API

2. Install the required packages:
   ```bash   
   pip install -r requirements.txt

4. Set up your MongoDB connection in *config/db.py* or in the *config.json* file.

5. Running the Application
   ```bash
   uvicorn main:app
