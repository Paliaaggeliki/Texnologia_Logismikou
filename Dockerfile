FROM python:3.11.9

WORKDIR C:\Users\user\OneDrive\Έγγραφα\texnologialogismikou\app.py

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
