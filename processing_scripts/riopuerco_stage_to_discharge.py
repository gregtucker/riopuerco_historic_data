#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
riopuerco_stage_to_discharge.py: calculate discharge from stage records.

@author: gtucker
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys


dh = 0.005  # stage adjustment increment, feet
verbose = False
plot_results = True
plot_to_file = True


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


def read_mean_daily_discharge(filename):
    """Read the USGS-estimated daily mean discharge values from a csv file.

    Assumes there is one column with decimal date/time, another with discharge
    or blank for dry.
    """
    import csv
    
    daily_mean_data = {}
    
    with open(filename, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=',')
        for row in myreader:
            if row[1] == '':
                val = 0.0
            else:
                try:
                    val = float(row[1])
                except:
                    val = 0.0
                    if verbose:
                        print(row[0], row[1])
            daily_mean_data[int(row[0])] = val
            
    return daily_mean_data


def display_days_data(datetime, stage, discharge):
    """Prints out the values for the day, for testing purposes."""
    for i in range(len(datetime)):
        print(str(datetime[i]) + ' ' + str(stage[i]) + ' ' + str(discharge[i]))
        print discharge[i]


def process_one_day_of_stage_records(datetime, gage_height, rt_gage_height,
                                     rt_discharge, usgs_mean_disch):
    """Calculate discharge from stage records for one day.
    
    Parameters
    ----------
    datetime : array of float
        List of numbers representing date and time for each record of the
        current day.
    gage_height : array of float
        List of gage height measurements, in feet (same length as datetime)
    rt_gage_height : array of float
        Gage heights in rating table
    rt_discharge : array of float
        Discharge values in rating table
    usgs_mean_disch : float
        Daily mean discharge for this day originally calculated by USGS
    """

    # Get the discharge corresponding to each stage measurement of the day,
    # by interpolating the rating table.
    q = np.interp(gage_height, rt_gage_height, rt_discharge)

    # Calculate the mean daily discharge that this discharge series represents
    qm_pred0 = calc_mean_daily_discharge(datetime, q)

    # Find the difference between this value and the original USGS value
    q_err0 = (usgs_mean_disch - qm_pred0) / usgs_mean_disch

    # If the error is less than 5%, we're good to go. More likely, however,
    # it will be off, so we will iterate on adjusting the stage shift until we
    # find a better match. This is a simple stepping algorith in which we
    # increase or decrease the stage measurements until the sign of the error
    # reverses, meaning we've crossed a minimum point.
    niter = 0
    h_adj = 0.0
    prev_q_err = q_err0
    best_adj = 0.0
    best_qm = qm_pred0
    best_abs_q_err = abs(q_err0)
    dir_adj = np.sign(q_err0)
    q_err = q_err0
    while abs(q_err)<=abs(prev_q_err) and niter<100:
        h_adj += dir_adj*dh
        q = np.interp(gage_height+h_adj, rt_gage_height, rt_discharge)
        qm_pred = calc_mean_daily_discharge(datetime, q)
        prev_q_err = q_err
        q_err = (usgs_mean_disch - qm_pred) / usgs_mean_disch
        if verbose:
            print 'adjustment =',h_adj,'  qm_pred =',qm_pred,'  q_err =',q_err
        if abs(q_err) < best_abs_q_err:
            best_qm = qm_pred
            best_abs_q_err = abs(q_err)
            best_adj = h_adj
        niter += 1
    
    # Since we probably overshot the best-fit, go back and interpolate again
    # with our best-fit adjustment.
    q = np.interp(gage_height + best_adj, rt_gage_height, rt_discharge)

    return q, best_qm, best_adj, best_abs_q_err


def write_revised_records_to_file(date_time_master, stage_master_original,
                                  stage_master_revised, discharge_master,
                                  usgs_mean_daily_discharge,
                                  mean_daily_discharge_master, filename):
    """Write date/time, stage, and discharge to a csv file."""
    with open(filename, 'w') as output_file:
        output_file.write('Date/time,Original gage height (ft),'
                          + 'Adjusted gage height (ft),'
                          + 'Discharge (cfs),'
                          + 'Reported mean daily discharge (cfs),'
                          + 'Calculated mean daily discharge (cfs)\n')
        for i in range(len(date_time_master)):
            output_file.write(str(date_time_master[i]) + ','
                              + str(stage_master_original[i]) + ','
                              + str(stage_master_revised[i]) + ','
                              + str(discharge_master[i]) + ','
                              + str(usgs_mean_daily_discharge[i]) + ','
                              + str(mean_daily_discharge_master[i]) + '\n')
    output_file.close()


def run(data_file_name, daily_mean_file_name, rating_table_dir, out_name,
        plot_path=''):
    """Run the full analysis.
    """
    
    # Read the data
    (date_time, stage, rt_filename) = read_hydrograph_data(data_file_name)
    
    # Read the daily mean discharge values
    mean_daily_discharge = read_mean_daily_discharge(daily_mean_file_name)

    # Initialize the search through the records
    first_rec_of_day = 0
    current_rec = 0

    # Create lists to hold today's date/time and stage records
    date_time_this_day = []
    stage_this_day = []

    # Create lists to hold all date/time and stage records
    date_time_master = []
    stage_master_original = []
    stage_master_revised = []
    discharge_master = []
    mean_daily_discharge_usgs = []
    mean_daily_discharge_master = []

    # If the first record isn't right at midnight, add a record for midnight,
    # assigning the same stage as the first record.
    if date_time[0] % 1 != 0.0:
        day_start = math.floor(date_time[0])
        date_time_this_day.append(day_start)
        stage_this_day.append(stage[0])
        #current_rec = 0  # back up one

    while current_rec < len(date_time):

        # Get today's day number
        this_day = math.floor(date_time[first_rec_of_day])
        if verbose:
            print('this_day ' + str(this_day))
        
        # Loop over successive records until we find one that falls on the next
        # day, or we hit the end of the records
        while (current_rec < len(date_time)
               and math.floor(date_time[current_rec]) == this_day):

            # Add the current record to this day
            date_time_this_day.append(date_time[current_rec])
            stage_this_day.append(stage[current_rec])

            # Advance the record
            current_rec += 1

        # Now, add a record for midnight between this day and the next... but
        # only if (a) no such midnight already exists, and (b) we're not at
        # the end of the file.
        if verbose:
            print(current_rec < len(date_time))
            if current_rec < len(date_time):
                print(date_time[current_rec])
                print(date_time[current_rec] % 1)
        if current_rec < len(date_time):
            if date_time[current_rec] % 1 != 0.0:
            
                # Interpolate stage between previous and next records
                dt0 = date_time[current_rec - 1]
                st0 = stage[current_rec - 1]
                dt1 = date_time[current_rec]
                st1 = stage[current_rec]
                midnight = math.floor(date_time[current_rec])
                m = (st1 - st0) / (dt1 - dt0)
                x = midnight - dt0
                stage_midnight = m * x + st0
                add_midnight_to_next = True

            else:
                midnight = date_time[current_rec]
                stage_midnight = stage[current_rec]
                add_midnight_to_next = False

            date_time_this_day.append(midnight)
            stage_this_day.append(stage_midnight)
            
        else:  # this means we're at the end of the file
            
            add_midnight_to_next = False  # so we won't have a next day
            if date_time[current_rec - 1] % 1 != 0.0:  # if last rec not midnight,
                date_time_this_day.append(this_day + 1.0)  # add midnight
                stage_this_day.append(stage[current_rec - 1])
                
        # Process this particular day

        # Get the name of the rating table file for this day
        (rt_gage_height, rt_discharge) = read_rating_table(rating_table_dir
            + '/' + rt_filename[first_rec_of_day])
        
        if verbose:
            print('About to process... day is:' + str(int(this_day)))
            print('mdd is' + str(mean_daily_discharge[int(this_day)]))
        if mean_daily_discharge[int(this_day)] == 0.0:
            q = []
            for i in range(len(date_time_this_day)):
                q.append(0.0)
            bqm = 0.0
            best_adj = 0.0
        else:
            # Process the day's stage records
            (q, bqm, best_adj, best_abs_q_err) = process_one_day_of_stage_records(
                    np.array(date_time_this_day), np.array(stage_this_day),
                    rt_gage_height, rt_discharge,
                    mean_daily_discharge[int(this_day)])

        # Add the data to the master lists
        for i in range(len(date_time_this_day)):
            date_time_master.append(date_time_this_day[i])
            stage_master_original.append(stage_this_day[i])
            stage_master_revised.append(stage_this_day[i] + best_adj)
            discharge_master.append(q[i])
            mean_daily_discharge_usgs.append(mean_daily_discharge[int(this_day)])
            mean_daily_discharge_master.append(bqm)

        if verbose:
            print('Finished a day:')
            display_days_data(date_time_this_day, stage_this_day, q)
        if plot_results:
            year = int(date_time_this_day[0] / 365.25)
            julian_day = np.array(date_time_this_day) - 365.25 * year
            this_julian_day = this_day - 365.25 * year
            plt.figure(1)
            plt.plot(julian_day, stage_this_day)
            plt.ylabel('Stage (ft)')
            plt.xlabel('Julian Day in ' + str(year + 1900))
            if plot_to_file:
                plt.savefig(plot_path + 'stage' + str(year + 1900) + '.png')
            plt.figure(2)
            plt.plot(julian_day, q)
            plt.xlabel('Julian Day in ' + str(year + 1900))
            plt.ylabel('Discharge (cfs)')
            mdd = mean_daily_discharge[int(this_day)]
            plt.plot([this_julian_day, this_julian_day + 1], [mdd, mdd])
            if plot_to_file:
                plt.savefig(plot_path + 'discharge' + str(year + 1900) + '.png')

        # Reset for the next
        first_rec_of_day = current_rec
        date_time_this_day = []
        stage_this_day = []
        if add_midnight_to_next:
            date_time_this_day.append(midnight)
            stage_this_day.append(stage_midnight)

    write_revised_records_to_file(date_time_master, stage_master_original,
                                  stage_master_revised, discharge_master,
                                  mean_daily_discharge_usgs,
                                  mean_daily_discharge_master,
                                  out_name)

if __name__ == '__main__':
    
    try:
        data_path = sys.argv[1]
        data_file_name = sys.argv[2]
        daily_mean_path = sys.argv[3]
        daily_mean_file = sys.argv[4]
        rating_table_dir = sys.argv[5]
        output_file_name = sys.argv[6]
        run(data_path + '/' + data_file_name,
            daily_mean_path + '/' + daily_mean_file, rating_table_dir,
            output_file_name)
        if plot_results:
            plt.show()
    except IndexError:
        print('\nUsage: python ' + sys.argv[0] + ' <data path> <data file name>'
              + ' <daily mean path> <daily mean name> <rating table path>'
              + ' <output file name>')
