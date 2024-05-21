FROM python:latest
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . .
CMD ["python", "main.py"]