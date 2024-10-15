import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import matplotlib





def plot_exp1(f_DAGP, f_DAGP_2, f_asy_dagp, f_asy_dagp_2, max_iter_syn, node_comp_time_exp1, \
               node_comp_time_exp1_2, neighbors, Delay_mat_dagp, Delay_mat_dagp_2, T_active_exp1, \
                T_active_exp1_2, current_dir, plot_iter=20000, save_results_folder='exp1', plot_time=43000, itrs=1750):

    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['pdf.fonttype'] = 42
    font = FontProperties()
    font.set_size(17)
    mark_every = 50000
    linewidth = 2

    os.chdir(current_dir)
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    plt.figure(1, figsize=(7, 5))
    plt.tick_params(labelsize=17, width=3)
    plt.plot(f_DAGP[:plot_iter], '-m', markevery = mark_every,linewidth = linewidth)
    plt.plot(f_DAGP_2[:plot_iter], '--m', markevery = mark_every,linewidth = linewidth)
    plt.plot(f_asy_dagp[:plot_iter],  '-y', markevery = mark_every,linewidth = linewidth)
    plt.plot(f_asy_dagp_2[:plot_iter],  '--y', markevery = mark_every,linewidth = linewidth)
    plt.legend([r'\textbf{DAGP}', r'\textbf{Throttled-DAGP}', r'\textbf{ASY-DAGP}', r'\textbf{ASY-Throttled-DAGP}'], prop={'size': 16})
    plt.xlabel(r'\textbf{Iterations}', fontsize=16)
    plt.ylabel(r'\textbf{Objective value}', fontsize=16)
    plt.grid(True)
    path = os.path.join(save_results_folder, 'iter')
    plt.savefig( path + ".pdf", format = 'pdf')

    T_sync = np.zeros(max_iter_syn+1)
    for i in range(1,max_iter_syn+1):
        tmp1 = node_comp_time_exp1[:,i]
        tmp2 = np.multiply(neighbors, Delay_mat_dagp[:,:,i])
        tmp3 = np.max(tmp2, axis=0)
        tmp4 = tmp1 + tmp3 
        tmp5 = np.max(tmp4)
        T_sync[i] = T_sync[i-1] + tmp5

    T_sync_2 = np.zeros(max_iter_syn+1)
    for i in range(1,max_iter_syn+1):
        tmp1 = node_comp_time_exp1_2[:,i]
        tmp2 = np.multiply(neighbors, Delay_mat_dagp_2[:,:,i])
        tmp3 = np.max(tmp2, axis=0)
        tmp4 = tmp1 + tmp3 
        tmp5 = np.max(tmp4)
        T_sync_2[i] = T_sync_2[i-1] + tmp5

    plt.figure(2, figsize=(7, 5))
    plt.tick_params(labelsize=17, width=3)
    plt.plot(T_sync[:itrs], f_DAGP[:itrs], '-m', markevery = mark_every,linewidth = linewidth)
    plt.plot(T_sync_2[:itrs], f_DAGP[:itrs], '--m', markevery = mark_every,linewidth = linewidth)
    plt.plot(T_active_exp1[:plot_time], f_asy_dagp[:plot_time], '-y', markevery = mark_every,linewidth = linewidth)
    plt.plot(T_active_exp1_2[:plot_time], f_asy_dagp_2[:plot_time], '--y', markevery = mark_every,linewidth = linewidth)
    plt.legend([r'\textbf{DAGP}', r'\textbf{Throttled-DAGP}', r'\textbf{ASY-DAGP}', r'\textbf{ASY-Throttled-DAGP}'], prop={'size': 16})
    plt.xlabel(r'\textbf{Time units}', fontsize=16)
    plt.ylabel(r'\textbf{Objective value}', fontsize=16)
    plt.grid(True)
    path = os.path.join(save_results_folder, 'time')
    plt.savefig( path + ".pdf", format = 'pdf')





def plot_exp2(T_active_exp2, res_F_asy_dagp, res_F_asyspa, res_F_appg, current_dir, save_results_folder, plot_iter=10050):
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['pdf.fonttype'] = 42
    font = FontProperties()
    font.set_size(17)
    font2 = FontProperties()
    font2.set_size(17)
    mark_every = 50000
    linewidth = 2

    os.chdir(current_dir)
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)
    plt.figure(4, figsize=(7, 5))
    plt.tick_params(labelsize=17, width=3)
    plt.plot(T_active_exp2[:plot_iter],   res_F_asy_dagp[:plot_iter], '-oy', markevery = mark_every,linewidth = linewidth)
    plt.plot(T_active_exp2[:plot_iter],   res_F_asyspa[:plot_iter],   '-ob', markevery = mark_every,linewidth = linewidth)
    plt.plot(T_active_exp2[:plot_iter],   res_F_appg[:plot_iter],     '-og', markevery = mark_every,linewidth = linewidth)
    plt.legend([r'\textbf{ASY-DAGP}', r'\textbf{ASY-SPA}', r'\textbf{APPG}'],  prop={'size': 16})
    plt.xlabel(r'\textbf{Time units}', fontsize=16)
    plt.ylabel(r'\textbf{Optimality gap}', fontsize=16)
    plt.yscale('log')
    plt.ylim( 10**-14, 1)
    plt.grid(True)
    path = os.path.join(save_results_folder, 'comparison_unconstianedd')
    plt.savefig( path + ".pdf", format = 'pdf')





def plot_exp3(T_active_exp3, f_asy_dagp, f_asy_pgex, current_dir, save_results_folder, plot_iter=5000):
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['pdf.fonttype'] = 42
    font = FontProperties()
    font.set_size(17)
    mark_every = 50000
    linewidth = 2

    os.chdir(current_dir)
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    plt.figure(1, figsize=(7, 5))
    plt.tick_params(labelsize=17, width=3)
    plt.plot(T_active_exp3[:plot_iter], f_asy_dagp[:plot_iter],  '-y', markevery = mark_every,linewidth = linewidth)
    plt.plot(T_active_exp3[:plot_iter], f_asy_pgex[:plot_iter],  '-r', markevery = mark_every,linewidth = linewidth)
    plt.legend([r'\textbf{ASY-DAGP}', r'\textbf{ASY-PG-EXTRA}'], prop={'size': 16})
    plt.xlabel(r'\textbf{Time Units}', fontsize=16)
    plt.ylabel(r'\textbf{Optimality Gap}', fontsize=16)
    plt.yscale('log')
    plt.ylim(10**-13,1)
    plt.grid(True)
    path = os.path.join(save_results_folder, 'dagp_vs_asy_pgExtra')
    plt.savefig( path + ".pdf", format = 'pdf')





def plot_exp4(T_active_exp4, f_asy_dagp0, f_asy_dagp1, f_asy_dagp2, f_asy_dagp3, current_dir, save_results_folder, plot_iter=7000):
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['pdf.fonttype'] = 42
    font = FontProperties()
    font.set_size(17)
    mark_every = 5000
    linewidth = 2

    os.chdir(current_dir)
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    plt.figure(3, figsize=(7, 5))
    plt.tick_params(labelsize=17, width=3)
    plt.plot(T_active_exp4[:plot_iter], f_asy_dagp0[:plot_iter],  '-oy', markevery = mark_every,linewidth = linewidth)
    plt.plot(T_active_exp4[:plot_iter], f_asy_dagp1[:plot_iter],  '-ok', markevery = mark_every,linewidth = linewidth)
    plt.plot(T_active_exp4[:plot_iter], f_asy_dagp2[:plot_iter],  '-or', markevery = mark_every,linewidth = linewidth)
    plt.plot(T_active_exp4[:plot_iter], f_asy_dagp3[:plot_iter],  '-oc', markevery = mark_every,linewidth = linewidth)
    plt.legend([r'\textbf{ASY-DAGP, $p=0.00$}', r'\textbf{ASY-DAGP, $p=0.25$}', r'\textbf{ASY-DAGP, $p=0.50$}', r'\textbf{ASY-DAGP, $p=0.75$}'], prop={'size': 16})
    plt.xlabel(r'\textbf{Time units}', fontsize=16)
    plt.ylabel(r'\textbf{Objective value}', fontsize=16)
    plt.grid(True)
    path = os.path.join(save_results_folder, 'drop_iter')
    plt.savefig( path + ".pdf", format = 'pdf')

