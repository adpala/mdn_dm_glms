import pandas as pd
import numpy as np
from scipy.signal import decimate
import sklearn.model_selection

bases_dict = {'window':600,'neye':0,'ncos':20,'kpeaks':(0,500),'b':40,'nbasis':20}

def get_delim(file_path):
    '''Determines separator used in csv table.'''
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
    if "," in first_line:
        delim = ","
    else:
        delim = " "
    return delim

def decimate_y(raw_y,decimating_values):
    '''Downsample vector by decimation.'''
    for dec_fold in decimating_values:
        raw_y = decimate(raw_y, dec_fold, ftype='fir')
    return raw_y

def load_into_pandas(dir_path='W:/apalaci/code/janache',neuron_list=['DN_SPEEDO','MDN','real_DNp17','imposter']):
    '''looks at all info xlsx tables from data folders, fixes filenames, adds ignore status, adds individual path to data, and fixes DN names'''

    # collect from all xlsx tables, expected in "./neuron/neuron.xlsx" structure, based on each neuron in neuron_list
    df = pd.concat([pd.read_excel(f'{dir_path}/{DN_type}/{DN_type}.xlsx').assign(**{'DN':DN_type, '#Location':'undefined'}) if '#Location' not in pd.read_excel(f'{dir_path}/{DN_type}/{DN_type}.xlsx').columns else pd.read_excel(f'{dir_path}/{DN_type}/{DN_type}.xlsx').assign(**{'DN':DN_type}) for DN_type in neuron_list]).reset_index(drop=True).rename(columns={'#File':'filename','#Location':'side'})
    
    # fix extra quotation marks in names
    df.loc[df['filename'].apply(lambda x: len(x)) == 17, 'filename'] = df['filename'][df['filename'].apply(lambda x: len(x)) == 17].apply(lambda x: x[1:-1])

    # add ignore status
    recordings_to_ignore = [
        "2022_06_09_0001","2022_06_24_0000","2022_06_24_0002","2022_06_28_0015",  # to ignore from first batch (after communication with Stefan)
        "2024_10_18_0003","2024_10_29_0007",'2024_10_29_0008',"2024_11_06_0005","2024_11_29_0009",  # problems in v_y
        "2024_11_28_0005", "2024_10_29_0002" # recordings with flight
        ]
    df['to_ignore'] = df['filename'].isin(recordings_to_ignore)

    # add absolute paths of each experiment to table
    df['abs_file_path'] = [f'{dir_path}/{neuron}/{filename}.csv' for filename, neuron in zip(df.filename, df.DN)]

    # rename DN
    df.loc[df.DN == 'DN_SPEEDO','DN'] = 'imposter'
    df.loc[df.DN == 'real_DNp17','DN'] = 'DNp17'

    return df

def load_recording(csv_path):
    '''Loads individual recording csv, renaming columns and calculating variable components (absolute, negative and positive).'''
    singleDN_df = pd.read_csv(csv_path,delimiter=get_delim(csv_path),header=None).rename(columns={0:'spike',1:'v_fwd',2:'v_ang',3:'v_y'})
    singleDN_df['abs_v_fwd'] = np.abs(singleDN_df['v_fwd'])
    singleDN_df['abs_v_ang'] = np.abs(singleDN_df['v_ang'])
    singleDN_df['abs_v_y'] = np.abs(singleDN_df['v_y'])
    singleDN_df['pos_v_fwd'] = np.clip(singleDN_df['v_fwd'],0,None)
    singleDN_df['pos_v_ang'] = np.clip(singleDN_df['v_ang'],0,None)
    singleDN_df['pos_v_y'] = np.clip(singleDN_df['v_y'],0,None)
    singleDN_df['neg_v_fwd'] = np.clip(singleDN_df['v_fwd'],None,0)
    singleDN_df['neg_v_ang'] = np.clip(singleDN_df['v_ang'],None,0)
    singleDN_df['neg_v_y'] = np.clip(singleDN_df['v_y'],None,0)
    return singleDN_df

def get_xy(singleDN_df,y_names,sample_frequency=20000,bin_width=100,decimating_values=[10,10]):
    '''Get corresponding non-zscored x and ys, binned by bin_width and decimated by decimating_values (respectively). Note: product of decimating_values should be equal to bin_width.'''
    time_raw = np.arange(len(singleDN_df['spike']))/sample_frequency
    x_raw = singleDN_df['spike'].fillna(0).copy()
    x_bins = np.arange(-bin_width,len(x_raw),bin_width)/sample_frequency
    x, _ = np.histogram(time_raw[x_raw==1],x_bins)
    ys = np.stack([decimate_y(singleDN_df[name].fillna(0).copy(),decimating_values) for name in y_names], axis=1)
    return x, ys

def chunked_test_train_split(X_b,y_m,block_size=5_000,n_block_min=5,test_size=0.35,random_state=42):
    '''Divides data into test-train split, implementing chunks of manageable size for better handling of memory, unless size is already small enough.'''
    if X_b.shape[0] > n_block_min*block_size:
        num_blocks = X_b.shape[0] // block_size
        X_b_chunked = X_b[:num_blocks * block_size].reshape((-1, block_size, X_b.shape[-1]))
        y_m_chunked = y_m[:num_blocks * block_size].reshape((-1, block_size, y_m.shape[-1]))

        blocks_train, blocks_test = sklearn.model_selection.train_test_split(np.arange(num_blocks), random_state=random_state, shuffle=True, test_size=test_size)

        X_train = np.concatenate(X_b_chunked[blocks_train])
        X_test = np.concatenate(X_b_chunked[blocks_test])
        y_train = np.concatenate(y_m_chunked[blocks_train])
        y_test = np.concatenate(y_m_chunked[blocks_test])
    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_b, y_m, random_state=random_state, shuffle=True, test_size=test_size)
    return X_train, X_test, y_train, y_test

def plot_glm_filter(ax,estimated_filters_dict,filename_list,varname,T,main_color='gray',with_l2=False,with_individual_traces=True,linestyle='-',lw=4,zero_line_lw=0.25):
    mean_filter_collection = None
    for filename in filename_list:
        if filename in estimated_filters_dict.keys():
            estimated_filters = estimated_filters_dict[filename][varname]

            if with_l2:
                l2_norms = np.linalg.norm(estimated_filters, ord=2, axis=1)
                mean_filter = np.nanmean(estimated_filters/l2_norms[:,np.newaxis],axis=0)
            else:
                mean_filter = np.nanmean(estimated_filters,axis=0)

            if mean_filter_collection is None:
                mean_filter_collection = mean_filter
            else:
                mean_filter_collection = np.vstack((mean_filter_collection, mean_filter))

            if with_individual_traces:
                ax.plot(T, mean_filter, color=main_color,alpha=0.2,lw=lw/8)
        else:
            print(f"{filename} missing")
    ax.axhline(y=0,color='k',lw=zero_line_lw)
    ax.plot(T, np.nanmean(mean_filter_collection,axis=0), color=main_color, lw=lw, linestyle=linestyle)