# import python image (same as env local)
FROM python:3.13-slim

# install libgomp1 for lightgbm to work
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# add a workdir dans le container
WORKDIR /app

# copy dependencies
COPY ./pyproject.toml /app/pyproject.toml

#fix bc Docker was looking for readme file
COPY pyproject.toml README.md /app/

# copy package
COPY ./src/ /app/src/

# install dependencines from toml fles
RUN pip install --upgrade pip \
    && pip install --no-cache-dir .


# run the project
CMD ["python", "src/entrypoint_bootstrap.py"]
