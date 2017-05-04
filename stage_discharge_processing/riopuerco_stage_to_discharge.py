#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
riopuerco_stage_to_discharge.py: calculate discharge from stage records.

@author: gtucker
"""

import numpy as np
import matplotlib.pyplot as plt
import math

dh = 0.005  # stage adjustment increment, feet
hydrograph_filename = 'rp_stage_time410501.csv'   # stage vs time, in ft and decimal days
qm = 414.0  # mean daily Q for this day, in cfs

def calc_mean_daily_discharge(t, q):
    """
    Uses the trapezoid method to compute mean discharge given a set of times
    and values.
    
    Parameters
    ----------
    t : array of float
        Times (in days) for discharge values
    q : array of float
        Discharge values (cfs)
        
    Example
    -------
    >>> import numpy
    >>> t = numpy.array([0., 2., 3., 8.])
    >>> q = numpy.array([2., 3., 5., 1.])
    >>> calc_mean_daily_discharge(t, q)
    3.0
    """
    n = len(t)
    assert(len(q)==n), 'unequal number of values for time and discharge'
    if n==1:  # if just one record, it IS the mean
        return q[0]
    dt = t[1:]-t[:n-1]          # time increment between readings
    qpair = (q[:n-1]+q[1:])/2   # average discharge between pairs
    total_time = t[-1] - t[0]   # elapsed time from beginning to end of record
    assert (total_time>0.0), 'hydrograph must have duration >0'
    return np.sum(dt*qpair)/total_time  # return the weighted average
    

def read_rating_table(filename):
    """Read stage and discharge for rating table from <filename>.
    
    Parameters
    ----------
    filename : string
        Name of comma-delimited text file containing rating table
        
    Returns
    -------
    tuple of float arrays
        stage, discharge values from table
    """
    rt = np.loadtxt(filename, delimiter=',')
    return np.array(rt[:,0]), np.array(rt[:,1])


def plot_rating_table(stage, discharge, output_file=None):
    """Plot discharge vs stage (assume units of cfs and ft)."""
    plt.plot(stage, discharge, '.-')
    plt.xlabel('Stage (ft)')
    plt.ylabel('Discharge (cfs)')
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
        
        
def get_rt_filenames(data):
    """From a 2D array with year-month-day start and end dates for each record,
    create names for rating table files.
    
    We assume that we're passed a 2D np array in which columns 0, 1, and 2 hold
    the year, month, and day for the start of the period of record for the
    rating table for a given record, and columns 3, 4, and 5 hold the end year
    month and day. We further assume that the rating table itself is contained
    in a file with a name formatted like (for example):
        
        rating_table_19520101to19530713.csv
    """
    rt_filenames = []
    for rec in range(data.shape[0]):
        date1 = (str(int(data[rec,0])) + str(int(data[rec,1])).zfill(2)
                 + str(int(data[rec,2])).zfill(2))
        date2 = (str(int(data[rec,3])) + str(int(data[rec,4])).zfill(2)
                 + str(int(data[rec,5])).zfill(2))
        rt_name = 'rating_table_' + date1 + 'to' + date2 + '.csv'
        rt_filenames.append(rt_name)

    return rt_filenames


def read_hydrograph_data(filename):
    """Read date/time, stage, and name of rating curve file from <filename>."""
    
    # Read the data from file
    hydro_data = np.loadtxt(filename, delimiter=',')

    # Convert the rating table start and end dates into a rating table filename
    rt_filenames = get_rt_filenames(hydro_data[:,2:])

    return hydro_data[:,0], hydro_data[:,1], rt_filenames


def run(data_file_name):
    """Run the full analysis.
    
    """
    
    # Read the data
    (date_time, stage, rt_filename) = read_hydrograph_data(data_file_name)

    # Initialize the search through the records
    first_rec_of_day = 0
    current_rec = 1

    # Create lists to hold today's date/time and stage records
    date_time_this_day = []
    stage_this_day = []

    # If the first record isn't right at midnight, add a record for midnight,
    # assigning the same stage as the first record.
    if date_time[0] % 1 != 0.0:
        day_start = math.floor(date_time[0])
        date_time_this_day.append(day_start)
        stage_this_day.append(stage[0])
        current_rec = 0  # back up one

    while current_rec < len(date_time):
        
        # Get today's day number
        this_day = math.floor(date_time[first_rec_of_day])
        print('this_day' + str(this_day))
        
        # Loop over successive records until we find one that falls on the next
        # day, or we hit the end of the records
        while (current_rec < len(date_time)
               and math.floor(date_time[current_rec]) == this_day):
        
            # Add the current record to this day
            date_time_this_day.append(date_time[current_rec])
            stage_this_day.append(stage[current_rec])

            # Advance the record
            current_rec += 1

        first_rec_of_day = current_rec
        print('Finished a day:')
        print(date_time_this_day)
        print(stage_this_day)
        date_time_this_day = []
        stage_this_day = []

        # SOME THINGS TO DO: IF FIRST REC OF NEW DAY IS MIDNIGHT, ADD IT, IF
        # NOT INTERPOLATE A MIDNIGHT

        # Now we've either hit the next day or the end of the file.
        
    
        # Is this midnight? If not, make a record for midnight
        #if date_time[first_reco_of_day] % 1 != 0.0:
            
            # Date-time of beginning of day (technically midnight prior day)
            #day_start = math.floor(date_time[first_rec_of_day])
            #date_time_this_day.append(day_start)
            
            # Interpolate to get stage...
            #  If this is the first record, set equal to first real record.
            #  Else, interpolate linearly between prior day and this one.
#            if first_rec_of_day == 0:
#                stage_this_day.append(stage[first_rec_of_day])
#            else:
#                # The good old fashioned formula for slope of a line:
#                # y = mx + b. Here b is the starting stage, m is change in 
#                # stage over time interval, and x is the time difference
#                # between last record of prior day and midnight. y of course is
#                # the interpolated stage.
#                prior_rec = first_rec_of_day - 1
#                b = stage[prior_rec]
#                m = ((stage[first_rec_of_day] - stage[prior_rec])
#                     / (date_time[first_rec_of_day] - date_time[prior_rec]))
#                x = day_start - date_time[prior_rec]  # time interval
#                stage_this_day.append((m * x) + b)
    
    
    #(rt_stage, rt_discharge) = read_rating_table(rating_table_filename)
    #plot_rating_table(rt_stage, rt_discharge)


if __name__ == '__main__':
    data_file_name = 'rp_rp_time_stage_rt_1953.csv'
    data_path = '/Users/gtucker/Data/RioPuerco/RioPuercoWork/riopuerco_historic_data/stage_discharge_processing'
    run(data_path + '/' + data_file_name)
    #read_hydrograph_data('/Users/gtucker/Data/RioPuerco/RioPuercoWork/riopuerco_historic_data/stage_discharge_processing/rp_rp_time_stage_rt_1953.csv')

