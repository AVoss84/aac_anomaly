FROM tiangolo/uwsgi-nginx-flask:python3.7
RUN apt-get clean -y && apt-get update -y
RUN apt-get install bash
RUN pip install --upgrade pip 
ENV UC_CODE_DIR /app/src
EXPOSE 5000
COPY requirements.txt requirements.txt
COPY . /app

RUN pip install -r requirements.txt
RUN pip install -e src

# Add config file to target folder
COPY config.toml /root/.streamlit/config.toml

WORKDIR /app
ENTRYPOINT [ "streamlit", "run" ]
CMD [ "app.py" ]