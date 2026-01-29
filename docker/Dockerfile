FROM jupyter/minimal-notebook:x86_64-ubuntu-22.04

USER ${NB_UID}

RUN pip install --no-cache-dir \
    --retries 5 \
    --timeout 100 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir \
    --retries 5 \
    --timeout 100 \
    -r /tmp/requirements.txt

WORKDIR /home/jovyan/work

EXPOSE 8888