# How to run this script:
# python3 main.py <sales_nb_dataset_file> 

import math, os, sys, time, vaex

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from calculate_population_knowing_the_product import get_population_knowing_product
from market_share_estimates import get_np_market_share_max
from seasonality_multipliers import get_seasonality_multipliers
from consumer_fidelity import get_fidelity_score
from awareness_score import get_awareness_score
from webscraping.webscraping_comments import webscrape_comments
from extract_features.extract_features import get_innovation_score

"""
Update the market share and sales numbers curves on the dashboard, 
depending on the selected DV (Distribution Value)
"""
def update_dashboard(population_knowing_the_product, dv, market_share_max, global_growth_rate, seasonality_multipliers, total_sales_numbers, displacement, innovation_score):
    # One point for each week of the 1st semester on the market
    market_share_per_week, sales_number_per_week = [], []
    for week in range(26):
        # Market share per week : Gompertz curve equation
        market_share_curr_week = (population_knowing_the_product["predicted"][week]/100) * market_share_max * math.exp(-displacement * math.exp(-global_growth_rate * week))
        market_share_per_week.append(market_share_curr_week)

    i = 0
    for _, seasonality in seasonality_multipliers.items(): 
        # Sales numbers per week
        if i < 26:
            market_share_curr_week = market_share_per_week[i]
            market_volume_curr = seasonality * total_sales_numbers
            sales_number_curr_week = market_share_curr_week * market_volume_curr * dv * seasonality * innovation_score
            sales_number_per_week.append(sales_number_curr_week)
            i += 1

    # Return predicted market shares and sales numbers of the new product
    data_points_ms = {'week': range(len(market_share_per_week)), 'predicted': market_share_per_week}
    data_points_sn = {'week': range(len(sales_number_per_week)), 'predicted': sales_number_per_week}
    print("-- Update dashboard DONE")
    return data_points_ms, data_points_sn

# Display the dashboard with predicted market shares and sales numbers curves depending on the week
def main(dv=''):
    # Define input variables to ask user what he wants to set up
    dataset = sys.argv[1]
    remove_bad_lines_file = "./remove_bad_lines_dataset.sh"
    ad_hoc = "./ranking.xlsx"
    pre_process_dataset = input("Pre-process dataset file [Y/n]: ")
    targeted_nb_of_contacts = int(input("Targeted nb of contacts: "))
    reach = float(input("Reach (between 0 and 1): "))
    frequency = float(input("Frequency (between 0 and 1): "))
    length_of_campaign = int(input("Length of campaign: "))
    dv = float(input("DV (between 0 and 1): "))
    nb_competing_products = int(input("Number of competing products: "))
    products_brand_to_webscrape = input("Product website to webscrape: ")
    product_type_to_webscrape = input("Product category to webscrape: ")
    nb_clusters_reviews = int(input("Number of clusters for products reviews: "))
    
    # Check if user typed dv value
    if dv == '':
        dv=0.1

    """ 
    Check if :
    - the sales numbers dataset exists
    - the file to pre-process the sales numbers dataset exists
    - the ad_hoc answers file exists
    """
    if not os.path.isfile(dataset) or not os.path.isfile(remove_bad_lines_file) or not os.path.isfile(ad_hoc):
        sys.exit("Dataset, remove bad lines file or ad-hoc file doesn't exist!")

    # Remove lines which contain more fields than scheduled
    if pre_process_dataset == 'Y':
        os.system(f"./{remove_bad_lines_file} {dataset}")

    # Create dataframe from the sales numbers dataset file
    headers = ['libelle_var','week','barcode','type','segment','category','description','weight','sales_number','price','sales_value','discount']
    dtypes = {
                'libelle_var': "category",
                'week': int,
                'barcode': int,
                'type': "category",
                'segment': "category",
                'category': "category",
                'description': "category",
                'weight': "category",
                'sales_number': int,
                'price': "category",
                'sales_value': "category",
                'discount': "category"
            }
    df = vaex.from_csv(dataset, sep=';', names=headers, index_col=False, encoding="utf-8", encoding_errors="ignore", dtype=dtypes)
    print("-- Create dataframe DONE")

    # Prepare the mathematical formula to predict market share
    population_knowing_the_product = get_population_knowing_product(targeted_nb_of_contacts, reach, frequency, length_of_campaign)
    displacement = 1
    visibility_score = float(get_awareness_score(dataset, ad_hoc))
    fidelity_score = float(get_fidelity_score(ad_hoc))
    market_share_max, prediction_error = get_np_market_share_max(df, ad_hoc, nb_competing_products)
    seasonality_multipliers, total_sales_numbers = get_seasonality_multipliers(df)
    global_growth_rate = visibility_score + fidelity_score
    webscrape_comments(products_brand_to_webscrape, product_type_to_webscrape)
    innovation_score = get_innovation_score('./webscraping/comments.txt', './extract_features/classification_model_weights.pth', nb_clusters_reviews)

    data_points_ms, data_points_sn = update_dashboard(population_knowing_the_product, dv, market_share_max, global_growth_rate, seasonality_multipliers, total_sales_numbers, displacement, innovation_score)

    return df, data_points_ms, data_points_sn, population_knowing_the_product, market_share_max, global_growth_rate, seasonality_multipliers, total_sales_numbers, displacement, innovation_score, prediction_error


if __name__ == '__main__':
    start_time = time.time()
    df, data_points_ms, data_points_sn, population_knowing_the_product, market_share_max, global_growth_rate, seasonality_multipliers, total_sales_numbers, displacement, innovation_score, prediction_error = main(0.1)
    end_time = time.time()
    final_time = end_time - start_time
    print(f"-- {final_time} seconds --")

    app = Dash(__name__)

    # Build the web dashboard
    app.layout = html.Div([
        html.H1('PFE-UC1 dashboard'),
        html.Div([
            dcc.Graph(id="graph")
        ]),
        html.P(f"Prediction error: {prediction_error}"),
        html.Div([
            html.P("DV (%):"),
            dcc.Dropdown(['10','20','30','40','50','60','70','80','90','100'], '10', id='dv-dropdown')
        ])
    ])

    @app.callback(
        Output("graph", "figure"), 
        Input("dv-dropdown", "value"))
    def update_figures(selected_dv):
        print(f"selected_dv={selected_dv}")
        start_time = time.time()
        data_points_ms, data_points_sn = update_dashboard(population_knowing_the_product,
                                                            float(selected_dv),
                                                            market_share_max,
                                                            global_growth_rate,
                                                            seasonality_multipliers,
                                                            total_sales_numbers, displacement, innovation_score)
        end_time = time.time()
        final_time = end_time - start_time
        print(f"-- {final_time} seconds --")

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add predicted market shares curve
        fig.add_trace(
            go.Scatter(x=list(data_points_ms['week']), y=list(data_points_ms['predicted']), name="Predicted market shares (%)"),
            secondary_y=False,
        )

        # Add predicted sales numbers curve
        fig.add_trace(
            go.Scatter(x=list(data_points_sn['week']), y=list(data_points_sn['predicted']), name="Predicted sales numbers"),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text="Predicted market shares and sales numbers"
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Week")

        # Set y-axes titles
        fig.update_yaxes(title_text="Predicted <b>market shares (%)</b>", secondary_y=False)
        fig.update_yaxes(title_text="Predicted <b>sales numbers</b>", secondary_y=True)

        return fig

    app.run_server(debug=True, use_reloader=False)