FROM python:3.12

WORKDIR /app

COPY fitness_app/ /app/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# CMD ["python", "app.py"]
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]