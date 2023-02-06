# WEBSCRAPING

# Goal : webscraping comments from Nocibe shampoo page to extract features
# we will only focus on nocibe website for the first tests 
# For the comments, any types of shampoo will be accepted to have the most comments for extraction
# When we find the specific nocibe page we want, we just add to the url a part which specify we want to have comments the most ranked
# The output is an array list of products with theirs consumers comments

from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time 
import numpy as np

def webscrape_comments(products_brand_to_webscrape, product_type_to_webscrape, min_nb_products_webscrape):
    #-- Websites to scrape
    # HERE : you can specify the website to webscrape, and the comment html id
    domain = {
          "domain":"https://www.nocibe.fr/cheveux/shampoing/C-47970",
          "url":"",
          "comment_id":"bv-content-summary-body-text",
          "item_comment_id":"bv-content-item bv-content-top-review bv-content-review",
          "class_products_menu":"proditem proditem__after-story",
          "class_name_products_menu":"proditem__infos-name",
          "next_page":""
      }

    # -- Google search website 
    google_serp = {
            #get url 
            "domain":"https://www.google.com/",
            "url":"",
            "website_id":{"class":"yuRUbf"},
            "website_html":"div"
        }
        
    # ---- CONSTANTS INITIALIZATION MAIN PAGE ----

    # Number of comments showed when you arrive on the product web page
    NUMBER_COMMENTS_SHOWED = 3

    # specification of nocibe shampoing webpage, it's to have the best product ratings so you avoid having products without comments
    lab_top_product_ratings="/TW-1055D_1054D_1060D#products"

    # List of url websites we want to scrape 
    website_url=[]
    google_serp["url"] = "https://www.google.com/search?q=site:{}+\"{}\"+{}&start=".format(domain["domain"], products_brand_to_webscrape, product_type_to_webscrape)
    list_comments_products = []

    # ---- WEBDRIVER INITIALIZATION ----

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280x1696")
    options.add_argument("--single-process")
    options.add_argument("--disable-dev-tools")
    options.add_argument("--no-zygote")
    options.add_argument("--remote-debugging-port=9222")   

    # open it, go to a website, and get results
    wdmain = webdriver.Chrome('chromedriver',options=options)
    wdmain.get(google_serp["url"])
    content = wdmain.page_source.encode('utf-8').strip()
    time.sleep(1)
    soup = BeautifulSoup(content,'html.parser')

    # we go on the google page and we look for the websites to scrape.
    # these sites are already chosen beforehand with the domain variable
    # here only the nocibé site is required
    websites_to_scrape = soup.find_all(google_serp["website_html"], google_serp["website_id"])
    time.sleep(5)

    for website in websites_to_scrape:
        if (domain["domain"] in str(website.find('a')['href'])):
            website_url = domain
            # we find the nocibe website + have the top product ratings 
            website_url.update({"url":domain["domain"]+lab_top_product_ratings})
        
    wdmain.get(website_url["url"])
    content = wdmain.page_source.encode('utf-8').strip()
    time.sleep(3)

    #Only with nocibé 
    # if the button is not showed because we have all the products , button_showed = 0
    button_showed=1
    soup = BeautifulSoup(wdmain.page_source,'html.parser')
    products = soup.find_all("div",{"class":domain["class_products_menu"]})

    # we want to click on the "show more products" button if it is showed and if we have less comments showed than expected ( =nb_products_to_webscrape)

    while button_showed == 1 and len(products) < min_nb_products_webscrape:
        try:
            # we try to find the button
            button = wdmain.find_element(By.CSS_SELECTOR,"#prodlist > div.prodlist__list > div.prodlist__list-wrap.prodlist__list-stories-1 > div.prodlist__loadmore > button")
            wdmain.execute_script("arguments[0].click()",button)
            time.sleep(1)
            soup = BeautifulSoup(wdmain.page_source, "html.parser")
            products = soup.find_all("div",{"class":domain["class_products_menu"]})
            # error if you don't find a show more button (you keep going without clicking on it)
        except AttributeError: 
            # the button does not exists, so we do not click and scrape every product of the main page
            button_showed=0
            continue

    soup = BeautifulSoup(wdmain.page_source, "html.parser")
    products = soup.find_all("div",{"class":domain["class_products_menu"]})

    # ---- WEBSCRAPING COMMENTS SHOWED ----
    count_product = 0
    for product in products:
        while count_product < min_nb_products_webscrape:
            count_product = count_product + 1
            # we try to scrape comments of a product
            # if we can't or if a problem happened, we skip to another product 
            try:
                name_product = product.find("strong",{"class":domain["class_name_products_menu"]}).get_text(strip=True)
                description_product = product.find("span",{"class":domain["class_name_products_menu"]}).get_text(strip=True)
                name_product = name_product + " | " + description_product
                    # it is the url product we retrieve from the menu
                url_product ="https://www.nocibe.fr"+product.find('a')['href'].strip()

                # --- LIST OF COMMENTS INITIALIZATION
                wdmain.get(url_product)

                #Only with Nocibé, it is a element that is loaded when you open a product page
                element = WebDriverWait(wdmain, 25).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="WAR"]'))).get_attribute("innerHTML")
                

                # ---- RETRIEVING PRODUCT COMMENTS ----

                soup = BeautifulSoup(wdmain.page_source,'html.parser')
                nb_search = wdmain.find_element(By.CLASS_NAME,"bv_numReviews_text").text

                # - We delete the parentheses because we get the number in the following form : (NUMBER)
                # and we just want to have the number

                characters = "()"
                for x in range(len(characters)):
                    nb_search = nb_search.replace(characters[x],"")

                # in the product webpage, we click on
                if int(nb_search) >= NUMBER_COMMENTS_SHOWED:
                    nb_click =(int(nb_search)-NUMBER_COMMENTS_SHOWED)/30 if (int(nb_search)-NUMBER_COMMENTS_SHOWED)%30 == 0 else (int(nb_search)-NUMBER_COMMENTS_SHOWED)/30 + 1
                else:
                    nb_click = 0

                for _ in range(int(nb_click)):
                    # CHANGE CSS SELECTOR OF THE BUTTON "SHOW MORE" IF IT IS NOT NOCIBE WEBPAGE
                    button =wdmain.find_element(By.CSS_SELECTOR,"#BVRRContainer > div > div > div > div > div.bv-content-pagination > div > button")
                    #button clicked
                    wdmain.execute_script("arguments[0].click()",button)
                    time.sleep(1)
                

                soup = BeautifulSoup(wdmain.page_source,'html.parser')

                # -- Webscraping comments showed

                cases = soup.find_all("li",{"class":domain["item_comment_id"]})

                # ---- RETRIEVING COMMENTS----

                for item in cases:
                    comment = item.find("div",{"class":domain["comment_id"]}).get_text(strip=True)
                    list_comments_products.append(comment)
            except:
                continue
    
    print("-- Webscraping : comments retrieving DONE")
    ar = np.array(list_comments_products) 
    np.savetxt('./webscraping/comments.txt', ar, delimiter = '\n', fmt='%s', encoding="utf-8")
    print("-- Webscraping : creation of comments file DONE")

    wdmain.quit()
