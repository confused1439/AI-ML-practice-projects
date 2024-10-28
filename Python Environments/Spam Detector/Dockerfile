# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Create a symlink so that 'python' points to 'python3'
# RUN ln -s /usr/local/bin/python3 /usr/local/bin/python

# Check if the symlink exists before creating it
RUN if [ ! -e /usr/local/bin/python ]; then ln -s /usr/local/bin/python3 /usr/local/bin/python; fi

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container as /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the application when the container launches
ENTRYPOINT [ "python", "./spam_predict.py" ]