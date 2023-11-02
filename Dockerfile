FROM python:3.10

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["test.py", "model.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "--timeout", "60", "--workers", "4", "--worker-class",  "gevent", "--worker-connections=1000", "--threads=3", "test:app" ]
