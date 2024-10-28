FROM python:3.12-slim

# Check if the symlink exists before creating it
RUN if [ ! -e /usr/local/bin/python ]; then ln -s /usr/local/bin/python3 /usr/local/bin/python; fi

# Set the wowrking dir in the container
WORKDIR /usr/src/app

# Copy the contain of current dir into the container as /usr/src/app
COPY . .

# Install the dependencies specified in the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose PORT 8080 for streamlit
EXPOSE 8080

# Set the streamlit configuration to run on port 8080
ENV STREAMLIT_SERVER_PORT=8080

# Run the streamlit app
CMD [ "streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false" ]