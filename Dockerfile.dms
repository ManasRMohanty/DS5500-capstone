FROM python:3.7

RUN apt-get update

RUN apt-get install -y -qq libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1

WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python3" ]

CMD ["scripts/server.py"]

