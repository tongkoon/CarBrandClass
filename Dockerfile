# 
FROM python:3.9

# 
WORKDIR /carbrand

# 
COPY ./requirements.txt /carbrand/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /carbrand/requirements.txt

# 
COPY ./app /carbrand/app
COPY ./model /carbrand/model

ENV PYTHONPATH "${PYTHONPATH}:/carbrand"

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]