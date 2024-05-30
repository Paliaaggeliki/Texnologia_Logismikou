FROM python:3.10.7

WORKDIR /Volumes/PALIA/3 ΕΤΟΣ/ΚΟΡΜΟΥ/ΤΕΧΝΟΛΟΓΙΑ ΛΟΓΙΣΜΙΚΟΥ/texnologia/app.py

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]