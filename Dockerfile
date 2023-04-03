FROM tiangolo/uwsgi-nginx-flask:python3.7
RUN apt-get clean -y && apt-get update -y
RUN apt-get install bash
RUN pip install --upgrade pip 
#EXPOSE 5000
EXPOSE 8080       
COPY . /app
RUN pip install -r requirements.txt
RUN pip install -e src
#RUN mkdir .streamlit 
#COPY config.toml .streamlit/config.toml     
WORKDIR /app
#HEALTHCHECK CMD curl --fail http://localhost:8080/health
CMD [ "streamlit", "run", "app.py", "--server.port=8080" ]
#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]