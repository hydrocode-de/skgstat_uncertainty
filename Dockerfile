ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}
LABEL maintainer='Mirko Mälicke'

# build the structure
RUN mkdir -p /src/skgstat_uncertainty
RUN mkdir -p /src/data

# copy the sources
COPY ./skgstat_uncertainty /src/skgstat_uncertainty

# COPY the packaging
COPY ./requirements.txt /src/requirements.txt
COPY ./setup.py /src/setup.py
COPY ./README.md /src/README.md

# copy the data
COPY ./data /src/data

# build the package
RUN pip install --upgrade pip
RUN cd /src && pip install -e .

# create the entrypoint
WORKDIR /src/skgstat_uncertainty/chapters
ENTRYPOINT ["streamlit", "run"]
CMD ["learn_variograms.py"]