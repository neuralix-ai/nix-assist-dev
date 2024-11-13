import pandas as pd
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_data():
    raw_data_path = os.getenv("RAW_DATA_PATH")
    files = [f for f in os.listdir(raw_data_path) if not f.startswith('.')]

    # Dynamically create DataFrames for each file
    dataframes = {f"df_{filename.split('.')[0]}": pd.read_csv(raw_data_path + filename) for filename in files}

    return dataframes

def clean_data(df):
    # Clean up column names
    df.columns = [re.sub(r"\s*\(.*?\)", "", col).strip().lower().replace(" ", "_") for col in df.columns]

    # Check for nulls and invalid entries and handle them
    # print(df.isna().sum())

    return df

def process_data(dict_df): 

    processed_data_path = os.getenv("PROCESSED_DATA_PATH")

    # dim_vehicles
    df_temp_vehicles = dict_df['df_vehicles'].rename(columns={
        'id': 'vehicle_id',
        'size': 'vehicle_size',
        'year': 'vehicle_purchase_year',
        'cost': 'vehicle_cost',
        'distance': 'vehicle_distance_bucket'
    })

    df_temp_vehicles_fuels = dict_df['df_vehicles_fuels'].rename(columns={ 
        'id': 'vehicle_id', 
        'consumption' : 'vehicle_fuel_consumption_rate'})

    # Perform a left join on 'ID' column
    df_dim_vehicles = pd.merge(
        df_temp_vehicles, 
        df_temp_vehicles_fuels, 
        how='left', 
        on='vehicle_id'
    )

    # dim_fuels
    df_dim_fuels = dict_df['df_fuels'].rename(columns={
        'year': 'year',
        'fuel': 'fuel',
        'emissions': 'fuel_emission_rate',
        'cost': 'cost_per_unit_fuel',
        'cost_uncertainty': 'fuel_cost_uncertainty_percentage'
    })

    # fact_demand
    df_fact_demand = dict_df['df_demand'].rename(columns={
    'distance': 'demand_distance_bucket',
    'size': 'demand_size_bucket'
    })

    # fact_carbon_emissions
    df_fact_carbon_emissions = dict_df['df_carbon_emissions'].rename(columns={
    'carbon_emission_co2/kg': 'target_carbon_emissions'
    })

    # dim_cost_profiles
    df_dim_cost_profiles = dict_df['df_cost_profiles'].rename(columns={
    'end_of_year': 'vehicle_age',
    'resale_value_%': 'resale_value_percentage',
    'insurance_cost_%': 'insurance_cost_percentage',
    'maintenance_cost_%': 'maintenance_cost_percentage'
    })  

    # fact_solution
    df_fact_solution = dict_df['df_solution'].rename(columns={
        'num_vehicles': 'number_of_vehicles',
        'type': 'action_type'
    })

    # Save processed data

    # dimensions
    df_dim_vehicles.to_csv(processed_data_path + 'dim_vehicles.csv')
    df_dim_fuels.to_csv(processed_data_path + 'dim_fuels.csv')
    df_dim_cost_profiles.to_csv(processed_data_path + 'dim_cost_profiles.csv')

    # facts
    df_fact_demand.to_csv(processed_data_path + 'fact_demand.csv')
    df_fact_carbon_emissions.to_csv(processed_data_path + 'fact_carbon_emissions.csv')
    df_fact_solution.to_csv(processed_data_path + 'fact_solution.csv')
    
def main():
    dict_df = get_data()
    dict_df = {name:clean_data(df) for name, df in dict_df.items()}

    process_data(dict_df)

if __name__ == "__main__":
    main()

