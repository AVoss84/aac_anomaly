FROM tiangolo/uwsgi-nginx-flask:python3.7
RUN apt-get clean -y && apt-get update -y
RUN apt-get install bash
RUN pip install --upgrade pip 
EXPOSE 5000
COPY . /app
RUN pip install -r requirements.txt
RUN pip install -e src
#RUN mkdir .streamlit 
#COPY config.toml .streamlit/config.toml     
WORKDIR /app
CMD [ "streamlit", "run", "app.py" ]