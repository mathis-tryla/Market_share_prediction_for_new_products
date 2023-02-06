FROM ubuntu

WORKDIR .

RUN apt-get update -y
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install python3-venv
RUN apt-get -y install net-tools curl nano
RUN apt-get -y install chromium-browser

COPY . /home/market_share_prediction_for_new_products/

RUN apt-get -y install /home/market_share_prediction_for_new_products/app/webscraping/google-chrome-stable_109.0.5414.119-1_amd64.deb

RUN chmod +x /home/market_share_prediction_for_new_products/app/remove_bad_lines_dataset.sh
RUN python3 -m pip --default-timeout=1000 install -r /home/market_share_prediction_for_new_products/requirements.txt

EXPOSE 8050
