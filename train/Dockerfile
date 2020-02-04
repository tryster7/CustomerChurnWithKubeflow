FROM python:3.7-slim
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir --requirement /tmp/requirements.txt
ADD train.py /
ENTRYPOINT [ "python3", "/train.py" ]
