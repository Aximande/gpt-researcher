# Use a specific version of Python on Debian Bullseye as the base image
FROM python:3.11.4-slim-bullseye as install-browser

# Update package list and install Chromium and Chromedriver
RUN apt-get update \
  && apt-get install -y --no-install-recommends chromium chromium-driver \
  && chromium --version \
  && chromedriver --version

# Install Firefox ESR and Geckodriver
RUN apt-get update \
  && apt-get install -y --fix-missing --no-install-recommends firefox-esr wget \
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

# Expose the port the app runs on and set the command to run the app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
