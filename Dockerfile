FROM ubuntu

WORKDIR .

RUN apt-get update -y
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install python3-venv
RUN apt-get -y install net-tools curl

COPY . /home/UC1/

RUN chmod +x /home/UC1/remove_bad_lines_shampoo.sh
RUN pip3 install -r /home/UC1/requirements.txt

EXPOSE 8050
