FROM ubuntu

WORKDIR .

RUN apt-get update -y
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install python3-venv
RUN apt-get -y install net-tools curl nano
RUN apt-get -y install chromium-browser

COPY . /home/market_share_prediction/

RUN chmod +x /home/market_share_prediction/app/remove_bad_lines_dataset.sh
RUN pip3 install -r /home/market_share_prediction/requirements.txt

EXPOSE 8050

ENTRYPOINT python3 /home/market_share_prediction/app/main.py /home/dataset/dataset.txt