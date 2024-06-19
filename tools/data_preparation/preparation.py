import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

from datetime import datetime

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.getLevelName(os.environ['LOG_LEVEL']))

class DataPreparation():
    def __init__(self, station_code:str, df_geo:pd.DataFrame) -> None:
        logging.info(f"Getting parameters for station code '{station_code}'")
        self.station_code = station_code
        self.station_name = df_geo[df_geo['station_code'] == station_code]['name'].iloc[0]
        self.latitude = df_geo[df_geo['station_code'] == station_code]['latitude'].iloc[0]
        self.longitude = df_geo[df_geo['station_code'] == station_code]['longitude'].iloc[0]
        self.height = df_geo[df_geo['station_code'] == station_code]['height'].iloc[0]
        self.comp_station_loaded = False
        self.flg_imputation = False
        self.null_values_report_history = [
            textwrap.dedent(f"""
            **GENERATED HISTORY REPORT FOR NULL VALUES**
            --------------------------------------------------------
            Station code:                                  {self.station_code}
            Station name:                                  {self.station_name}
            --------------------------------------------------------"""),
        ]
        self.null_values_count = 0

    def load_data_csv(self, directory:str='', sep:str=',', header:int=0) -> None:
        try:
            self.file_path = os.path.join(directory, f"{self.station_code}.csv")
            logging.info(f"Reading file from local directory '{self.file_path}'")
            self.df_raw = pd.read_csv(self.file_path, sep=sep, header=header)
            logging.info(f"File {self.station_code}.csv was successfully read from directory!")
        except Exception as e:
            logging.error(f"There was an error while trying to read file from directory {self.file_path}")
            raise e
        
        logging.info("Updating timestamp's data type to 'datetime64' using format '%Y-%m-%d %H:%M:%S'")
        self.df_raw['timestamp'] = pd.to_datetime(self.df_raw['timestamp'], format='%Y-%m-%d %H:%M:%S')
        logging.debug(f"\n{self.df_raw.dtypes}")

    def __update_null_history_report(self, title:str='Report'):
        self.null_values_count += 1
        self.null_values_report_history.append(textwrap.dedent(f"""
        **{self.null_values_count}. {title.upper()} - {datetime.now()}**                                                       
        --------------------------------------------------------
        Total rows of the dataset:                     {self.df_raw.shape[0]}
        Total rows with null values:                   {self.df_null_values.shape[0]}
        --------------------------------------------------------
        Number of dates with null values:              {self.df_dates_null.shape[0]}
        Number of dates with just 1 null value:        {self.df_dates_null[self.df_dates_null['nulls_count'] == 1].shape[0]}
        Number of dates with 2 null values:            {self.df_dates_null[self.df_dates_null['nulls_count'] == 2].shape[0]}
        Number of dates with 3 or more null values:    {self.df_dates_null[self.df_dates_null['nulls_count'] >= 3].shape[0]}
        --------------------------------------------------------"""))

    def find_null_values(self, columns:list, generate_report:bool=True, report_title:str=None):
        logging.info(f"Looking for null values in dataset columns {columns}...")
        if self.flg_imputation:
            self.df_null_values = self.df_station[self.df_station[columns].isna().any(axis=1)]
        else:
            self.df_null_values = self.df_raw[self.df_raw[columns].isna().any(axis=1)]
        logging.warn(f"There were found {self.df_null_values.shape[0]} rows with null values!")
        self.df_dates_null = self.df_null_values\
            .groupby(self.df_null_values.timestamp.dt.date)\
            .count()[['timestamp']]\
            .rename(columns={'timestamp': 'nulls_count'})\
            .reset_index()\
            .sort_values(by='timestamp', ascending=True)
        
        if generate_report:
            if report_title:
                self.__update_null_history_report(title=report_title)
            else:
                self.__update_null_history_report()

        logging.info(''.join(self.null_values_report_history))

    def compare_stations(self, comp_station_code:str, start_date:str, end_date:str, meteorological_vars:list, 
                         sep:str=',', header:int=0):
        if ~self.comp_station_loaded:
            try:
                self.comp_station_code = comp_station_code
                comp_file_path = os.path.join(os.path.dirname(self.file_path), f"{comp_station_code}.csv")
                logging.info(f"Reading file from local directory '{comp_file_path}'")
                self.df_comp = pd.read_csv(comp_file_path, sep=sep, header=header)
                logging.info(f"File {comp_station_code}.csv was successfully read from directory!")
            except Exception as e:
                logging.error(f"There was an error while trying to read file from directory {comp_file_path}")
                raise e
        
            logging.info("Updating timestamp's data type to 'datetime64' using format '%Y-%m-%d %H:%M:%S'")
            self.df_comp['timestamp'] = pd.to_datetime(self.df_comp['timestamp'], format='%Y-%m-%d %H:%M:%S')
            logging.debug(f"\n{self.df_comp.dtypes}")
            self.comp_station_loaded = True

        logging.info("Starting stations comparison...")
        logging.info(f"Comparing stations {self.station_code} and {comp_station_code}...")
        self.__plot_variables(comp_station_code, start_date, end_date, meteorological_vars)

    def __plot_variables(self, comp_station_code:str, start_date:str, end_date:str, 
                         meteorological_vars:list):
        fig, ax = plt.subplots(len(meteorological_vars), 1, figsize=(12,15))
        for index, i in enumerate(meteorological_vars):
            self.df_raw[self.df_raw.timestamp.between(start_date, end_date)]\
                .plot(x='timestamp', y=i, c='#CB0006', linestyle='-', marker='.', 
                      alpha=0.4 if self.flg_imputation else 0.8, label=f"{i} ({self.station_code})", ax=ax[index])
            self.df_comp[self.df_comp.timestamp.between(start_date, end_date)]\
                .plot(x='timestamp', y=i, c='#0E3D84', linestyle='--', marker='x', 
                      alpha=0.4 if self.flg_imputation else 0.8, label=f"{i} ({comp_station_code})", ax=ax[index])
            if self.flg_imputation:
                self.df_station[self.df_station.timestamp.between(start_date, end_date)]\
                    .plot(x='timestamp', y=i, c='#1F1F1F', linestyle='', marker='.', 
                        alpha=0.9, label=f"{i} ({self.station_code} UPDATED)", ax=ax[index])
            ax[index].set_ylabel(i)
            if index == len(meteorological_vars)-1:
                ax[index].set_xlabel("Marca de tiempo")
            else:
                ax[index].set_xlabel(None)
            ax[index].grid(linestyle=':')
        plt.show()

    def data_imputation(self, meteorological_vars:list, interpolate:bool=False, interpolation_limit:int=None):
        try:
            logging.info(f"Filling '{self.station_code}' null values by using available data from station '{self.comp_station_code}'")
            logging.info(f"Merging dataframes from {self.station_code} and {self.comp_station_code}")
            df_merged = self.df_raw.merge(right=self.df_comp, on='timestamp', how='inner')
            logging.debug(f"\n{df_merged.dtypes}")
            logging.info("Filling null values...")
            for var in meteorological_vars:
                df_merged[f"{var}_x"] = df_merged[f"{var}_x"].fillna(df_merged[f"{var}_y"])
            self.df_station = df_merged[['timestamp'] + [f"{i}_x" for i in self.df_raw.columns if i != 'timestamp']]\
                .rename(columns={f"{i}_x": i for i in self.df_raw.columns if i != 'timestamp'})
        except Exception as e:
            logging.error(f"There was a problem while trying to fill {self.station_code} null values by using {self.comp_station_code}")
            raise e
        
        if interpolate:
            logging.info(f"Interpolation is enabled")
            if interpolation_limit:
                logging.info(f"Interpolation was configured for a maximum of {interpolation_limit} consecutive null values")
            else:
                logging.warn(f"There's no interpolation limit configured!")

            self.df_station.set_index('timestamp', inplace=True)
            self.df_station.interpolate(method='time', limit=interpolation_limit, limit_direction='both', inplace=True)
            self.df_station.reset_index(inplace=True)

        self.flg_imputation = True

        self.find_null_values(meteorological_vars, report_title='Data imputation report')

        return self.df_station