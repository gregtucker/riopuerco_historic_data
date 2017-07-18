#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
run_rio_puerco_1940s.py

Run the stage-to-discharge processing script on Rio Puerco stage records from
the 1940s, and specifically for years: ...

@author: gtucker
"""

import riopuerco_stage_to_discharge as rp

data_path = '../input_data_sheets/'

daily_mean = ('../mean_daily_discharge/'
              + 'rio_puerco_at_rio_puerco_mean_daily_discharge'
              + '19340301-19761231.csv')

rating_table_dir = '../rating_tables'

plot_path = '../plots/'

# 1944
rp.run(data_path + 'rio_puerco_at_rio_puerco_date_time_stage_1944.csv',
       daily_mean, rating_table_dir,
       '../data_files/rio_puerco_at_rio_puerco_stage_discharge1944.csv',
       plot_path)

# 1947
rp.run(data_path + 'rio_puerco_at_rio_puerco_date_time_stage_1947.csv',
       daily_mean, rating_table_dir,
       '../data_files/rio_puerco_at_rio_puerco_stage_discharge1947.csv',
       plot_path)

