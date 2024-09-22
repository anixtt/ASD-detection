import os
import re
import numpy as np
import csv

from nilearn.connectome import ConnectivityMeasure

def calculateCorrelation(dataset_path, corr_matrics_dir,
                         pcorr_matrics_dir, avg_pcorr_filepath, labels_filepath,
                         Tdc_path, Asd_path):

    # define the path of dataset and related file path
    dataset_path = dataset_path
    corr_matrices_dir = corr_matrics_dir
    pcorr_matrices_dir = pcorr_matrics_dir
    avg_pcorr_file = avg_pcorr_filepath
    labels_file = labels_filepath
    Tdc_path = Tdc_path
    Asd_path = Asd_path

    # create file path
    if not os.path.exists(corr_matrices_dir):
        os.makedirs(corr_matrices_dir)
    if not os.path.exists(pcorr_matrices_dir):
        os.makedirs(pcorr_matrices_dir)

    # each rois correlation
    corr_measure = ConnectivityMeasure(kind='correlation')
    # rois partial correlation
    pcorr_measure = ConnectivityMeasure(kind='partial correlation')


    # Load time series data(ROI_SIGNALS)
    # Get all Tdc subjects ID from each preprocessed file name
    Tdc_files = [os.path.join(Tdc_path, file) for file in os.listdir(Tdc_path)]
    Tdc_filename_list = []
    Tdc_subjectID_list = []
    Tdc_time_series = [0]*len(Tdc_files)
    for file in Tdc_files:
        Tdc_filename = file.split('/')[-1].split('.')[0]
        Tdc_filename_list.append(Tdc_filename)
        subjectID = re.findall("00[0-9]+", Tdc_filename)[0]
        Tdc_subjectID_list.append(subjectID)
    # Get Tdc time series data
    for i in range(len(Tdc_files)):
        Tdc_time_series[i] = np.loadtxt(Tdc_files[i])

    # Get all Asd subjects ID from each preprocessed file name
    Asd_files = [os.path.join(Asd_path, file) for file in os.listdir(Asd_path)]
    Asd_filename_list = []
    Asd_subjectID_list = []
    Asd_time_series = [0]*len(Asd_files)
    for file in Asd_files:
        Asd_filename = file.split('/')[-1].split('.')[0]
        Asd_filename_list.append(Asd_filename)
        subjectID = re.findall("00[0-9]+", Asd_filename)[0]
        Asd_subjectID_list.append(subjectID)
    # Get Asd time series data
    for i in range(len(Asd_files)):
        Asd_time_series[i] = np.loadtxt(Asd_files[i])

    # generate labels
    labels = [0]*(len(Asd_subjectID_list) + len(Tdc_subjectID_list))
    for i in range(len(Asd_subjectID_list)):
        labels[i] = {"subjectID": Asd_filename_list[i], "label": 1}
    for i in range(len(Tdc_subjectID_list)):
        labels[i + len(Asd_subjectID_list)] = \
            {"subjectID": Tdc_filename_list[i], "label": 0}
    with open(labels_file, 'w') as f:
        w = csv.writer(f)
        for i in labels:
            w.writerow(i.values())

    # Fit the correlation and partial correlation estimator
    #  to the given time series for each Asd subject
    Asd_corr_matrices = corr_measure.fit_transform(Asd_time_series)
    Asd_pcorr_matrices = pcorr_measure.fit_transform(Asd_time_series)

    # Save correlation and partial correlation matrices
    for i in range(0, len(Asd_corr_matrices)):
        if not os.path.exists(f'{corr_matrices_dir}/'+Asd_filename_list[i]+'.csv'):
            np.savetxt(f'{corr_matrices_dir}/'+Asd_filename_list[i]+'.csv', Asd_corr_matrices[i], delimiter=',')
        if not os.path.exists(f'{pcorr_matrices_dir}/' + Asd_filename_list[i] + '.csv'):
            np.savetxt(f'{pcorr_matrices_dir}/'+Asd_filename_list[i]+'.csv', Asd_pcorr_matrices[i], delimiter=',')

    # Fit the correlation and partial correlation estimator
    #  to the given time series for each Tdc subject
    Tdc_corr_matrices = corr_measure.fit_transform(Tdc_time_series)
    Tdc_pcorr_matrices = pcorr_measure.fit_transform(Tdc_time_series)

    # Save correlation and partial correlation matrices
    for i in range(0, len(Tdc_corr_matrices)):
        if not os.path.exists(f'{corr_matrices_dir}/'+Tdc_filename_list[i]+'.csv'):
            np.savetxt(f'{corr_matrices_dir}/'+Tdc_filename_list[i]+'.csv', Tdc_corr_matrices[i], delimiter=',')
        if not os.path.exists(f'{pcorr_matrices_dir}/' + Tdc_filename_list[i] + '.csv'):
            np.savetxt(f'{pcorr_matrices_dir}/'+Tdc_filename_list[i]+'.csv', Tdc_pcorr_matrices[i], delimiter=',')

    # Get average partial correlation matrix across time series and save
    All_time_series = Tdc_time_series + Asd_time_series
    All_pcorr_matrices = pcorr_measure.fit_transform(All_time_series)
    avg_pcorr_matrix = np.mean(All_pcorr_matrices, axis=0)
    np.savetxt(avg_pcorr_file, avg_pcorr_matrix, delimiter=',')

if __name__ == '__main__':
    rois = ['rois_cc200', 'rois_cc400']
    # define the path of each variables
    dataset_path = '../dataset'
    for roi in rois:
        corr_matrices_dir = f'{dataset_path}/FCmatrics/dparsf/filt_global/'+roi+'/corr_matrices'
        pcorr_matrices_dir = f'{dataset_path}/FCmatrics/dparsf/filt_global/'+roi+'/pcorr_matrices'
        avg_pcorr_file = f'{dataset_path}/FCmatrics/dparsf/filt_global/'+roi+'/avg_pcorr.csv'
        labels_file = f'{dataset_path}/FCmatrics/dparsf/filt_global/'+roi+'/labels.csv'
        Tdc_path = f'{dataset_path}/Tdc/Outputs/dparsf/filt_global/'+roi
        Asd_path = f'{dataset_path}/Asd/Outputs/dparsf/filt_global/'+roi
        # call the calculateCorrelation function
        calculateCorrelation(dataset_path, corr_matrices_dir, pcorr_matrices_dir,
                             avg_pcorr_file, labels_file, Tdc_path, Asd_path)
        print("FC matrices are calculated and saved.")