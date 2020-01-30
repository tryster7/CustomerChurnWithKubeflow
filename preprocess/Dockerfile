FROM python:3.7-slim
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt
ADD preprocess.py /
ENTRYPOINT [ "python", "./preprocess.py" ]
