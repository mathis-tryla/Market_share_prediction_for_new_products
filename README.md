# Market share prediction for new products
## _IG2i-Centrale Lille students 2023_

Here is the documentation dedicated to the end-of-study project realized by the following IG2i-Centrale Lille students from 2022 to 2023 :
- TRYLA Mathis
- FREMAUX Martin
- EL ATLASSI Nada
- SHAHPAZIAN Nicolas
- ROBERT Jules

The aim of this project was to **evaluate the market share of a new product** before it hits the market and predict the quantities to be produced and delivered in terms of logistic.

## Install dependencies

First you have to install python packages by using `pip` command with the `requirements.txt` file :

```sh
cd market_share_prediction
pip install -r requirements.txt
```

## Request GPT-3

In order to request GPT3 for the features extraction, you have to generate an OpenAI secret key :
1. Go to https://openai.com/api/
2. Click on _LOG IN_
3. Authenticate yourself with a Microsoft or a Google account
4. At the top-right corner of the webpage, click on _Personal_ and then on _View API keys_
5. Click on _Create a new secret key_ and copy the generated API key


## Docker

The solution is easy to install and deploy in a Docker container.
Here is how you simply build the docker image :

```sh
cd market_share_prediction
docker build -t market_share_prediction_docker_image .
```

> Note: It is required to install Docker on your local machine before beginning this Docker part.

This will create the market_share_prediction_docker_image image and pull in the necessary dependencies.

Once done, run the Docker image. Here is what you have to type on your terminal :

```sh
docker run -it -p 8050:8050 -v <dataset_directory_path>:/home market_share_prediction_docker_image
```

## Run the program

Finally, you can run our market share prediction program as follows :

```sh
cd /home/market_share_prediction
python3 main.py
```

Once done, you'll be asked to answer few questions. One of these questions is to paste the generated OpenAI API secret key (generated during the _Request GPT-3_ part of this documentation). 

Finally, you can verify the deployment by navigating to your server address in
your preferred browser.

```sh
http://127.0.0.1:8050
```

You'll be able to interact with our dashboard.
