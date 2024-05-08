# Use a specific version of Python on Debian Bullseye as the base image
FROM python:3.11.4-slim-bullseye as install-browser

# Update package list and install necessary dependencies
RUN apt-get update && apt-get install -y \
  gnupg \
  libnss3-dev \
  ca-certificates \
  fonts-liberation \
  wget \
  unzip

# Add the Google Chrome repository and install the latest stable version of Chromium
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list
RUN apt-get update && apt-get install -y google-chrome-stable

# Download and extract the appropriate version of Chromedriver
ENV CHROMEDRIVER_VERSION 113.0.5672.63
RUN mkdir -p /chromedriver \
  && wget -q --continue -P /chromedriver "http://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip" \
  && unzip /chromedriver/chromedriver* -d /chromedriver

# Add the Chromedriver binary to the system's PATH
ENV PATH="/chromedriver:${PATH}"

# Install Firefox ESR and Geckodriver
RUN apt-get update \
  && apt-get install -y --no-install-recommends firefox-esr \
  && wget https://github.com/mozilla/geckodriver/releases/download/v0.33.0/geckodriver-v0.33.0-linux64.tar.gz \
  && tar -xvzf geckodriver-v0.33.0-linux64.tar.gz \
  && chmod +x geckodriver \
  && mv geckodriver /usr/local/bin/ \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

# Prepare the installation environment
FROM install-browser as gpt-researcher-install

ENV PIP_ROOT_USER_ACTION=ignore

RUN mkdir /usr/src/app
WORKDIR /usr/src/app

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prepare the application environment
FROM gpt-researcher-install AS gpt-researcher

RUN useradd -ms /bin/bash gpt-researcher \
  && chown -R gpt-researcher:gpt-researcher /usr/src/app

USER gpt-researcher

COPY --chown=gpt-researcher:gpt-researcher ./ ./

# Set the command to run the app
CMD ["python", "main.py"]
