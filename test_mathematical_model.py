# How to run this script :
# 1: python3 mathematical_model.py visibility_score(float) fidelity_score(float) monthly_growth_rate(file.csv) market_share_max(float)
# Ex: python3 mathematical_model.py 0.1 0.0875 monthly_growth_rate.csv 25

import math, os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from calculate_population_knowing_the_product import get_population_knowing_product
from market_share_estimates import get_np_market_share_max
from shampoos_seasonality_multipliers import get_shampoos_seasonality_multipliers
from consumer_fidelity import get_fidelity_score
from visibility_score import get_visibility_score

def show_model_simulation(entity_per_week, type):
    x = np.array(range(26))
    y = entity_per_week

    plt.plot(x, y, "r", label="Predicted")
    plt.title("The evolution of the market share of the new product")
    plt.xlabel("time (week)")
    if type == "ms":
        plt.ylabel("market share (%)")
    elif type == "sn":
        plt.ylabel("sales number")
    plt.legend(loc="lower right")
    plt.show()

def export_to_csv(data_points, type):
    df_points = pd.DataFrame(data=data_points)
    df_points.set_index('week', inplace=True)
    csv_file = "points.csv"
    if type == "ms":
        csv_file = "points_market_shares.csv"
    elif type == "sn":
        csv_file = "points_sales_numbers.csv"
    df_points.to_csv(csv_file)

def update_dashboard(population_knowing_the_product, dv, market_share_max, global_growth_rate, shampoos_seasonality_multipliers, total_sales_numbers, displacement):
    print("-- Update dashboard DONE")
    # one point for each week of the 1st semester on the market
    market_share_per_week, sales_number_per_week = [], []
    for week in range(26):
        print(week)
        # Gompertz curve equation
        market_share_curr_week = (population_knowing_the_product["predicted"][week]/100) * market_share_max * math.exp(-displacement * math.exp(-global_growth_rate * week))
        market_share_per_week.append(market_share_curr_week)
    # Plot market shares and sales number of the new product per week
    # ms = market share
    # sn = sales number
    #show_model_simulation(market_share_per_week, "ms")

    i = 0
    for _, seasonality in shampoos_seasonality_multipliers.items(): 
        # Sales numbers per week
        # Vente(i) = part de marché(i) * volume du marché(i) * DV * coeff de saisonnalité(i)
        market_share_curr_week = market_share_per_week[i]
        market_volume_curr = seasonality * total_sales_numbers
        sales_number_curr_week = market_share_curr_week * market_volume_curr * dv * seasonality 
        sales_number_per_week.append(sales_number_curr_week)
        i += 1
    #show_model_simulation(sales_number_per_week[:26], "sn")

    # export points to csv
    data_points_ms = {'week': range(len(market_share_per_week)), 'predicted': market_share_per_week}
    data_points_sn = {'week': range(len(sales_number_per_week)), 'predicted': sales_number_per_week}
    export_to_csv(data_points_ms, "ms")
    export_to_csv(data_points_sn, "sn")
    return data_points_ms, data_points_sn

def main(dv=''):
    dataset = input("Shampoos dataset file: ")
    remove_bad_lines_file = input("Remove bad lines file: ")
    ad_hoc = input("Ad-hoc file: ")
    targeted_nb_of_contacts = int(input("Targeted nb of contacts: "))
    reach = float(input("Reach: "))
    frequency = float(input("Frequency: "))
    length_of_campaign = int(input("Length of campaign: "))
    dv = float(input("DV (between 0 and 1): "))
    """dataset="../SHAMPOING_UC1.txt"
    dataset="../SHAMPOING_UC1_026.txt"
    remove_bad_lines_file="../remove_bad_lines_shampoo.sh"
    ad_hoc="../ranking.xlsx"
    targeted_nb_of_contacts=3
    reach=0.25
    frequency=0.5
    length_of_campaign=10"""
    if dv == '':
        dv=0.3

    if not os.path.isfile(dataset) or not os.path.isfile(remove_bad_lines_file) or not os.path.isfile(ad_hoc):
        sys.exit("Dataset, remove bad lines file or ad-hoc file doesn't exist!")

    # Create dataframe
    headers = ['libelle_var','week','barcodes','type','segment','category','description','weight','sales_numbers','price','sales_value','discounts']
    df = pd.read_csv(dataset, sep=';', names=headers, index_col=False, encoding="utf-8", encoding_errors="ignore")
    print("-- Create dataframe DONE")

    # Remove lines which contains more fields than scheduled
    os.system(f"./{remove_bad_lines_file} {dataset}")

    population_knowing_the_product = get_population_knowing_product(targeted_nb_of_contacts, reach, frequency, length_of_campaign)
    displacement = 1
    visibility_score = float(get_visibility_score(dataset, ad_hoc))
    fidelity_score = float(get_fidelity_score(ad_hoc))
    market_share_max = get_np_market_share_max(df, ad_hoc)
    shampoos_seasonality_multipliers, total_sales_numbers = get_shampoos_seasonality_multipliers(df)
    global_growth_rate = visibility_score + fidelity_score

    data_points_ms, data_points_sn = update_dashboard(population_knowing_the_product, dv, market_share_max, global_growth_rate, shampoos_seasonality_multipliers, total_sales_numbers, displacement)

    return df, data_points_ms, data_points_sn, population_knowing_the_product, market_share_max, global_growth_rate, shampoos_seasonality_multipliers, total_sales_numbers, displacement


start_time = time.time()
df, data_points_ms, data_points_sn, population_knowing_the_product, market_share_max, global_growth_rate, shampoos_seasonality_multipliers, total_sales_numbers, displacement = main(0.1)
end_time = time.time()
final_time = end_time - start_time

print(f"-- Calculate population knowing the product DONE")
print(f"-- {final_time} seconds --")

app = Dash(__name__)

min_week = int(df['week'].values.min())
max_week = int(df['week'].values.max())

app.layout = html.Div([
    html.H1('PFE-UC1 dashboard'),
    html.Div([
        dcc.Graph(id="graph")
    ]),
    html.Div([
        html.P("DV (%):"),
        dcc.Dropdown(['10','20','30','40','50','60','70','80','90','100'], '10', id='dv-dropdown'), 
        html.P("Week:"),
        dcc.Slider(
            id='slider-week', min=min_week, max=max_week, 
            value=min_week, step=1)
    ])
])

@app.callback(
    Output("graph", "figure"), 
    Input("dv-dropdown", "value"),
    Input("slider-week", "value"))
def update_figures(selected_dv, selected_week):
    print(f"selected_dv={selected_dv}")
    start_time = time.time()
    data_points_ms, data_points_sn = update_dashboard(population_knowing_the_product,
                                                        float(selected_dv),
                                                        market_share_max,
                                                        global_growth_rate,
                                                        shampoos_seasonality_multipliers,
                                                        total_sales_numbers, displacement)
    end_time = time.time()
    final_time = end_time - start_time

    print(f"-- Calculate population knowing the product DONE")
    print(f"-- {final_time} seconds --")

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=list(data_points_ms['week']), y=list(data_points_ms['predicted']), name="Predicted market shares (%)"),
        secondary_y=False,
    )

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
