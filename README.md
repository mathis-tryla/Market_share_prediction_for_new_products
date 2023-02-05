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

## Set up sales numbers dataset file

In order to set up the sales numbers dataset file, you have to fill it in by writing values in this order :

```python3
['libelle_var','week','barcode','type','segment','category','description','weight','sales_number','price','sales_value','discount']
```

> Note: One row from this file corresponds to one sale.

Each field is separated from `;`


## Docker

The solution is easy to install and deploy in a Docker container.
Here is how you simply build the docker image :

```sh
cd market_share_prediction/app

docker build -t market_share_prediction_img ../
```

> Note: It is required to install Docker on your local machine before beginning this Docker part.

This will create the market_share_prediction_docker_image image and pull in the necessary dependencies.

Once done, run the Docker image. Here is what you have to type on your terminal :

```sh
docker run -it -p 8050:8050 -v <dataset_directory_path>:/home/dataset market_share_prediction_img
```

## Run the program

Finally, you can run our market share prediction program as follows :

```sh
cd /home/market_share_prediction/app

python3 main.py /home/dataset/dataset.txt
```

Once done, you'll be asked to answer few questions. One of these questions is to paste the generated OpenAI API secret key (generated during the _Request GPT-3_ part of this documentation). 

Finally, you can verify the deployment by navigating to your server address in
your preferred browser.

```sh
http://127.0.0.1:8050
```

You'll be able to interact with our dashboard.