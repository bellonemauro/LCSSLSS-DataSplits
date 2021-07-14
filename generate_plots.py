import sys
import os
import os.path
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results_folder = './results/'

tot_sets = 10
valid_sets = 10

def get_matrix_results(filename, results_folder):
    matrix = np.zeros((tot_sets,4))
    rows = np.ones((tot_sets))*1000
    with open(results_folder+filename+'.txt','r') as txt:
        for i, line in enumerate(txt.readlines()):
            temp = line.rstrip('\n').split(' ')
            rows[i] = temp[0]
            matrix[i,0] = temp[1]
            matrix[i,1] = temp[2]
            matrix[i,2] = temp[3]
            matrix[i,3] = temp[4]
    sort_idx = np.argsort(rows)
    matrix = matrix[sort_idx[:valid_sets],:]
    return matrix



def plot_data(day_fair, day_rain, night_fair, night_rain, modality):
    _, axs = plt.subplots(nrows=1,ncols=4,figsize=(13, 4))
    #fig.subplots_adjust(wspace=0)
    #fig.subplots_adjust(hspace=0.2)
    y_upper_lim = 92
    y_bottom_lim = 60
    ######################################
    fix_data(day_fair,axs,0)
    axs[0].set_ylabel('$\Delta$IoU [%]')
    axs[0].set_title('Day - Fair')
    axs[0].set_ylim(y_bottom_lim,y_upper_lim)

    fix_data(day_rain,axs,1)
    #axs[1].set_ylabel('IoU [%]')
    axs[1].set_title('Day - Rain')
    axs[1].set_ylim(y_bottom_lim,y_upper_lim)

    fix_data(night_fair,axs,2)
    #axs[2].set_ylabel('IoU [%]')
    axs[2].set_title('Night - Fair')
    axs[2].set_ylim(y_bottom_lim,y_upper_lim)

    fix_data(night_rain,axs,3)
    #axs[3].set_ylabel('IoU [%]')
    axs[3].set_title('Night - Rain')
    axs[3].set_ylim(y_bottom_lim,y_upper_lim)

    plt.savefig('./results/img/results_{}.pdf'.format(modality),bbox_inches='tight')


def fix_data(data,axs,cl):
    data_reduced = data #- data[:,1:2]

    df = pd.DataFrame(data_reduced, columns=['Base','Cotrain','Fusion','Upper'])
    df.head()

    vals, names, xs = [],[],[]
    for i, col in enumerate(df.columns):
        vals.append(df[col].values)
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

    axs[cl].boxplot(vals, labels=names,notch=False,showfliers=False)
    palette = ['r', 'g', 'b', 'y']
    for x, val, c in zip(xs, vals, palette):
        axs[cl].scatter(x, val, alpha=0.4, color=c)

    axs[cl].grid(which='minor', alpha=0.2)
    axs[cl].grid(which='major', alpha=0.5)

    axs[cl].axhline(y=np.mean(data_reduced[:,0]), color='r', linestyle='--', linewidth=1)
    axs[cl].axhline(y=np.mean(data_reduced[:,1]), color='g', linestyle='--', linewidth=1)
    axs[cl].axhline(y=np.mean(data_reduced[:,2]), color='b', linestyle='--', linewidth=1)
    axs[cl].axhline(y=np.mean(data_reduced[:,3]), color='y', linestyle='--', linewidth=1)


def convert_data(baseline,cotrain,semi,upper):
    day_fair = np.concatenate((baseline[:,0:1], cotrain[:,0:1],semi[:,0:1],upper[:,0:1]),axis=1)
    day_rain = np.concatenate((baseline[:,1:2], cotrain[:,1:2],semi[:,1:2],upper[:,1:2]),axis=1)
    night_fair = np.concatenate((baseline[:,2:3], cotrain[:,2:3],semi[:,2:3],upper[:,2:3]),axis=1)
    night_rain = np.concatenate((baseline[:,3:4], cotrain[:,3:4],semi[:,3:4],upper[:,3:4]),axis=1)

    return day_fair, day_rain, night_fair, night_rain

super_rgb = get_matrix_results('super_rgb_log',results_folder)
super_lidar = get_matrix_results('super_lidar_log',results_folder)
super_fusion = get_matrix_results('super_fusion_log',results_folder)

baseline_rgb = get_matrix_results('baseline_rgb_log',results_folder)
baseline_lidar = get_matrix_results('baseline_lidar_log',results_folder)
baseline_fusion = get_matrix_results('baseline_fusion_log',results_folder)

semi_rgb = get_matrix_results('semi_rgb_log',results_folder)
semi_lidar = get_matrix_results('semi_lidar_log',results_folder)
semi_fusion = get_matrix_results('semi_fusion_log',results_folder)

cotrain_rgb = get_matrix_results('cotrain_rgb_log',results_folder)
cotrain_lidar = get_matrix_results('cotrain_lidar_log',results_folder)
cotrain_fusion = get_matrix_results('cotrain_fusion_log',results_folder)

upper_rgb = get_matrix_results('upperbound_rgb_log',results_folder)
upper_lidar = get_matrix_results('upperbound_lidar_log',results_folder)
upper_fusion = get_matrix_results('upperbound_fusion_log',results_folder)


day_fair_rgb, day_rain_rgb, night_fair_rgb, night_rain_rgb = convert_data(baseline_rgb,cotrain_rgb,semi_rgb,upper_rgb)
day_fair_lidar, day_rain_lidar, night_fair_lidar, night_rain_lidar = convert_data(baseline_lidar,cotrain_lidar,semi_lidar,upper_lidar)
day_fair_fusion, day_rain_fusion, night_fair_fusion, night_rain_fusion = convert_data(baseline_fusion,cotrain_fusion,semi_fusion,upper_fusion)

plot_data(day_fair_rgb, day_rain_rgb, night_fair_rgb, night_rain_rgb, 'rgb')
plot_data(day_fair_lidar, day_rain_lidar, night_fair_lidar, night_rain_lidar, 'lidar')
plot_data(day_fair_fusion, day_rain_fusion, night_fair_fusion, night_rain_fusion, 'fusion')

##RGB
mean_day_fair_rgb = np.nanmean(day_fair_rgb,axis=0)
std_day_fair_rgb = np.nanstd(day_fair_rgb,axis=0)
mean_day_rain_rgb = np.nanmean(day_rain_rgb,axis=0)
std_day_rain_rgb = np.nanstd(day_rain_rgb,axis=0)
mean_night_fair_rgb = np.nanmean(night_fair_rgb,axis=0)
std_night_fair_rgb = np.nanstd(night_fair_rgb,axis=0)
mean_night_rain_rgb = np.nanmean(night_rain_rgb,axis=0)
std_night_rain_rgb = np.nanstd(night_rain_rgb,axis=0)

##lidar
mean_day_fair_lidar = np.nanmean(day_fair_lidar,axis=0)
std_day_fair_lidar = np.nanstd(day_fair_lidar,axis=0)
mean_day_rain_lidar = np.nanmean(day_rain_lidar,axis=0)
std_day_rain_lidar = np.nanstd(day_rain_lidar,axis=0)
mean_night_fair_lidar = np.nanmean(night_fair_lidar,axis=0)
std_night_fair_lidar = np.nanstd(night_fair_lidar,axis=0)
mean_night_rain_lidar = np.nanmean(night_rain_lidar,axis=0)
std_night_rain_lidar = np.nanstd(night_rain_lidar,axis=0)

##fusion
mean_day_fair_fusion = np.nanmean(day_fair_fusion,axis=0)
std_day_fair_fusion = np.nanstd(day_fair_fusion,axis=0)
mean_day_rain_fusion = np.nanmean(day_rain_fusion,axis=0)
std_day_rain_fusion = np.nanstd(day_rain_fusion,axis=0)
mean_night_fair_fusion = np.nanmean(night_fair_fusion,axis=0)
std_night_fair_fusion = np.nanstd(night_fair_fusion,axis=0)
mean_night_rain_fusion = np.nanmean(night_rain_fusion,axis=0)
std_night_rain_fusion = np.nanstd(night_rain_fusion,axis=0)


def print_table_part(mean, std):
       a,b = mean[0],std[0]
       c,d,e, = mean[1], std[1], mean[1] - mean[0]
       f,g,h = mean[2], std[2],mean[2] - mean[0]
       k,l,m = mean[3], std[3],mean[3] - mean[0]
       print('${:.2f}\pm{:.2f}$ & ${:.2f}\pm {:.2f} \hspace{{1mm}}({:.2f})$  & ${:.2f}\pm {:.2f} \hspace{{1mm}}({:.2f})$  & ${:.2f}\pm {:.2f} \hspace{{1mm}}({:.2f})$'.format(a,b,c,d,e,f,g,h,k,l,m)) 

def print_table_section(mean_rgb,std_rgb,mean_lidar,std_lidar,mean_fusion,std_fusion):
    print_table_part(mean_rgb,std_rgb)
    print_table_part(mean_lidar,std_lidar)
    print_table_part(mean_fusion,std_fusion)
    print('-----------------------------------------------------------------------------------------------------')

print_table_section(mean_day_fair_rgb,std_day_fair_rgb,mean_day_fair_lidar,std_day_fair_lidar,mean_day_fair_fusion,std_day_fair_fusion)
print_table_section(mean_day_rain_rgb,std_day_rain_rgb,mean_day_rain_lidar,std_day_rain_lidar,mean_day_rain_fusion,std_day_rain_fusion)
print_table_section(mean_night_fair_rgb,std_night_fair_rgb,mean_night_fair_lidar,std_night_fair_lidar,mean_night_fair_fusion,std_night_fair_fusion)
print_table_section(mean_night_rain_rgb,std_night_rain_rgb,mean_night_rain_lidar,std_night_rain_lidar,mean_night_rain_fusion,std_day_rain_fusion)
