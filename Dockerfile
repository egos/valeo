FROM python:3.9-slim

RUN mkdir /sl-valeo

COPY main.py /sl-valeo
COPY utils.py /sl-valeo
COPY requirements.txt /sl-valeo
COPY data.xlsx /sl-valeo


WORKDIR /sl-valeo
RUN pip install -r requirements.txt

EXPOSE 8501


CMD streamlit run main.py

