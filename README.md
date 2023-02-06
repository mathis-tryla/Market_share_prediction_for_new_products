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
git clone https://github.com/Warrior62/Market_share_prediction_for_new_products.git

cd market_share_prediction_for_new_products

pip install -r requirements.txt
```

## Set up sales numbers dataset file

In order to set up the sales numbers dataset file, you have to fill it in by writing values in this order :

```python3
['libelle_var','week','barcode','type','segment','category','description','weight','sales_number','price','sales_value','discount']
```

> Note: One row from this file corresponds to one sale.

Each field is separated from `;`


## Set up ad-hoc answers document

In order to fill in the ad-hoc answers document, you have to fill in the app/ranking.xlsx file with the following first row :

| Header fields  | Possible responses | Response examples  |
|---|---|---|
| Quel est votre sexe ? [Homme/Femme] | Homme;Femme| Femme |
| Quel âge avez-vous ?  | [14 ans ou moins;15-29 ans;30-44 ans;45-59 ans;60-74 ans;75 ans ou plus] | 15-29 ans  |   
| Quelle est votre tranche de salaire annuelle brute ?  | [Moins de 20 434€;20 435€ - 25 345€;25 346 € - 31 621€;31 621 € - 40 807€;40 808€ - 65 156€] | 20 435€ - 25 345€ |
| Connaissez-vous chacun des produits concurrents déjà commercialisés ? [Oui/Non] | [Oui;Non] | Oui |
| Classez le produit étudié et ces <n> produits en fonction de votre volonté d’achat (ranking) entre 1 et <n> [prod<n>]  | [1,2,...,n] | 2 |
| A quelle fréquence souhaitez-vous racheter ce produit dans le futur ? | [1 fois toutes les 2 semaines;1 fois par mois.;Au plus une fois tous les 2 mois.] | 1 fois par mois.  |
| Sur une échelle de 1 à 5 à quel point recommanderiez-vous ce produit à vos proches ? | [1;2;3;4;5] | 4  |
| Notez le rapport qualité-prix du produit entre 1 et 5 | [1;2;3;4;5] | 5  |
| Notez le visuel de l'emballage du produit entre 1 et 5 | [1;2;3;4;5] | 3  |
| Combien achetez vous de produits différents issus de cette catégorie en 1 an? | [0:+infinite] | 5  |


## Docker

The solution is easy to install and deploy in a Docker container.
Here is how you simply build the docker image :

```sh
cd market_share_prediction_for_new_products/app

docker build -t market_share_prediction_for_new_products_img ../
```

> Note: It is required to install Docker on your local machine before beginning this Docker part.

This will create the market_share_prediction_for_new_products_img image and pull in the necessary dependencies.

Once done, run the Docker image. Here is what you have to type on your terminal :

```sh
docker run -it -p 8050:8050 -v <dataset_directory_path>:/home/dataset market_share_prediction_for_new_products_img
```

## Run the program

Finally, you can run our market share prediction program as follows :

```sh
cd /home/market_share_prediction_for_new_products/app

python3 main.py /home/dataset/<dataset.txt>
```

## Answer questions
Once done, you'll be asked to answer a few questions:

| Questions  | Response examples  |
|---|---|
| Pre-process dataset file [Y/n]  |  n |
| Targeted nb of contacts  | 3  |   
| Reach (between 0 and 1)  | 0.25  |
| Frequency (between 0 and 1)  | 0.5  |
| Length of campaign  | 10  |
| DV (between 0 and 1)  | 0.7  |
| Number of competing products  | 5  |
| Product website to webscrape  | Nocibé  |
| Product category to webscrape  | Shampoing  |
| Number of products to webscrape  | 3  |
| Number of clusters for products reviews  | 12  |   

## Verify the deployment
Finally, you can verify the deployment by navigating to your server address in
your preferred browser on your local machine:

```sh
http://127.0.0.1:8050
```

You'll be able to interact with our dashboard.
