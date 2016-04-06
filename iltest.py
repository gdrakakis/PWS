#!flask/bin/python

from __future__ import division
from flask import Flask, jsonify, abort, request, make_response, url_for
import json
import pickle
import base64
import numpy
import math
import scipy
from copy import deepcopy
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn import linear_model
from numpy  import array, shape, where, in1d
import ast
import threading
import Queue
import time
import random
from random import randrange
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import io
from io import BytesIO
import cStringIO
from numpy import random
import scipy
from scipy.stats import chisquare
from copy import deepcopy
import operator 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from operator import itemgetter
#from PIL import Image ## Hide for production
app = Flask(__name__, static_url_path = "")


"""
    JSON Parser for interlabtest
"""
def getJsonContents (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        predictionFeature = jsonInput["predictionFeature"]
        parameters = jsonInput["parameters"]

        dataEntry = dataset.get("dataEntry", None)

        variables = dataEntry[0]["values"].keys() 

        owners_labs = []
        target_variable_values = []
        target_var_with_replicates = [] ## replicates
        uncertainty_values = []
        for i in range(len(dataEntry)):
            owners_labs.append(dataEntry[i]["compound"].get("ownerUUID"))
            for j in variables:		
                temp = dataEntry[i]["values"].get(j)
                temp_list = deepcopy(temp)
                if isinstance (temp, list):
                    for k in range (len(temp)):
                        temp[k] = float(temp[k])
                        temp_list[k] = float(temp_list[k]) ## replicates
                        temp_list[k] = round(temp_list[k], 2) ## replicates
                    temp = numpy.average(temp)
                    temp = round(temp, 2)
                else:
                    try:
                        if isinstance (float(temp), float):
                            temp = float(temp)
                            temp = round(temp, 2)
                    except:
                        pass

                if j == predictionFeature: #values == "predictionFeature"
                    target_variable_values.append(temp)
                    target_var_with_replicates.append(temp_list) ## replicates
                else:
                    uncertainty_values.append(temp)
        
        #uncertainty_values = [] # test case 
        if uncertainty_values == []:
            for i in range (len(target_variable_values)):
                uncertainty_values.append(0)
        """
        data_list = [[],[],[]]
        data_list[0] = owners_labs
        data_list[1] = target_variable_values
        data_list[2] = uncertainty_values
        """
        data_list = []
        data_list.append(owners_labs)
        data_list.append(target_variable_values)
        data_list.append(uncertainty_values)

        data_with_replicates = [] ## replicates
        data_with_replicates.append(owners_labs) 
        data_with_replicates.append(target_var_with_replicates) ## replicates
        data_with_replicates.append(uncertainty_values) 

    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"

    #data_list = sorted(data_list, key=itemgetter(1)) 
    data_list_transposed = map(list, zip(*data_list)) 
    data_with_replicates_transposed = map(list, zip(*data_with_replicates)) 
    return data_list_transposed, data_list, data_with_replicates_transposed, data_with_replicates # report in reverse # with replicates


"""
    LOCAL for csv and txt files - Get data. Returns list AND list_transposed
"""
def get_data ():
    path = "C:/Users/Georgios Drakakis/Desktop/Table8Unsorted.csv"
    #path = "C:/Users/Georgios Drakakis/Desktop/Table2.txt"
    
    # 1 =file has column names, 0 = no, just data
    has_header = 1

    # "," || "\t"
    delimitter = "," 
    #delimitter = "\t" 

    dataObj = open(path)
    my_list = []

    if has_header == 1:
        header = dataObj.readline()
        header = header.strip()
        header = header.split(delimitter)    

    while 1:
        line = dataObj.readline()
        if not line:
            break
        else:
            line = line.strip()
            line = line.replace("'", "") 
            temp = line.split(delimitter)

        for i in range (1,len(temp)):
            temp[i] = eval(temp[i])
            if isinstance (temp[i], list):
                for j in range (len(temp[i])):
                    temp[i][j] = float(temp[i][j])
                temp[i] = numpy.average(temp[i])
            else:
                try:
                    if isinstance (float(temp[i]), float):
                        temp[i] = float(temp[i])
                except:
                    pass
        #print temp
        my_list.append(temp)
    #print my_list
    # my_list = [ [Lab, Value, Uncertainty], [Lab, Value, Uncertainty], ...]
    # my_list_transposed = [ [Labs], [Values], [Uncertainties] ], sorted by Values
    my_list = sorted(my_list, key=itemgetter(1)) 
    my_list_transposed = map(list, zip(*my_list)) 

    return header, my_list, my_list_transposed 

"""
    initialise robust average and standard deviation
    *before* Algorithm A
"""
def init_robust_avg_and_std (sorted_data_list):
    x_star = numpy.median(sorted_data_list)
    median_list = abs(sorted_data_list - x_star)
    s_star =  1.483*numpy.median(median_list)
    x_star = round (x_star, 2)
    s_star = round (s_star, 2)
    return x_star, s_star

"""
    Algorithm A
"""
def algorithm_a(sorted_data_list, x_star, s_star):
    delta = 1.5*s_star
    new_sorted_data_list = []
    for i in range (len(sorted_data_list)):
        if sorted_data_list[i] < x_star - delta:
            new_sorted_data_list.append(x_star - delta)
        elif (sorted_data_list[i] > x_star + delta):
            new_sorted_data_list.append(x_star + delta)
        else: 
            new_sorted_data_list.append(sorted_data_list[i])
    new_sorted_data_list = numpy.array(new_sorted_data_list)
    x_star_new = sum(new_sorted_data_list)/len(new_sorted_data_list)
    s_star_new = 1.134*numpy.sqrt( (sum(numpy.power (new_sorted_data_list - x_star_new,2 ))) / (len(new_sorted_data_list) - 1) )
    x_star_new = round(x_star_new,2) 
    s_star_new = round(s_star_new,2)
    return x_star_new, s_star_new, new_sorted_data_list

"""
    Loop Algorithm A until values converge
"""
def loop_algorithm_a (sorted_data_list):
    x,s = init_robust_avg_and_std(sorted_data_list)
    tempx, temps = x,s
    temp_list = deepcopy(sorted_data_list)
    while 1:
        new_tempx, new_temps, new_temp_list = algorithm_a (temp_list, tempx, temps)
        if new_tempx == tempx and new_temps == temps: 
            break
        else:
            tempx = new_tempx
            temps = new_temps
            temp_list = deepcopy(new_temp_list)
    return new_temp_list, tempx, temps # rename????

"""
    Get standard uncertainty of assigned value from expert labs
"""
def get_uncertainty_for_assigned_value(s, p, uncertainties = []):
    #print "\n\n\n", s,p,uncertainties, "\n\n\n"
    if uncertainties != []:
        U_X_assigned = (1.25 / p) * numpy.sqrt(sum(numpy.power(uncertainties,2))) 
    else:
        U_X_assigned = (1.25 * s) / numpy.sqrt(p)

    U_X_assigned = round (U_X_assigned,2)
    return U_X_assigned

"""
    Removes data likely to make plots unreadable
"""
def kill_outliers(dataTransposed, robust_avg_x, robust_std_s):
    temp = [[],[],[]]
    for i in range (len(dataTransposed[1])):
        if dataTransposed[1][i] < robust_avg_x + 3.5* robust_std_s and dataTransposed[1][i] > robust_avg_x - 3.5* robust_std_s:
            temp[0].append(dataTransposed[0][i])
            temp[1].append(dataTransposed[1][i])
            temp[2].append(dataTransposed[2][i])
    dataTransposed = deepcopy(temp)
    return dataTransposed


"""
     Create histograms for report (on raw data)
"""
def hist_plots(num_bins, my_list_transposed, header):

    bins = numpy.linspace(min(my_list_transposed[1]), max(my_list_transposed[1]), num_bins+1)
    #print "BINS--->", bins
    labels = []
    for i in range (num_bins):
        labels.append("")

    for i in range (len(my_list_transposed[1])):
        for j in range (num_bins):
            if my_list_transposed[1][i] >= bins[j] and my_list_transposed[1][i] <= bins[j+1]:
                labels[j] = labels[j] + "    " + str(my_list_transposed[0][i])

    colours = []
    #print num_bins
    for i in range (num_bins):
        colours.append(random.rand(3,1)) # random colour scheme
                

    #myFIGA1 = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') ###
    myFIGA1a = plt.figure()

    # hist returns: num inst, bin bound, patches
    nn, bb, pp  = plt.hist(my_list_transposed[1], bins=num_bins, histtype='barstacked', color = random.rand(3,1), stacked=True) # , normed=True
    ##print bb
    ## copy labels for 2nd histogram (first copy gets altered here)
    labels_copy = deepcopy(labels) 

    for i in range (len(pp)-1,0,-1):
        if labels[i] == "":
            labels.pop(i)
            pp.pop(i)
            colours.pop(i)
    for i in range (len(pp)):
        plt.setp(pp[i], color=colours[i])  

    ##################################################################################################
    # extra space for legend
    #leg_box = plt.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2,  mode="expand", borderaxespad=0.,fontsize = 'x-small')
    #myFIGA.savefig('C:/Python27/inter_test1.png', dpi=300, format='png', bbox_extra_artists=(leg_box,), bbox_inches='tight')
    #
    # local maximize window
    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())
    ##################################################################################################
	
    """
    myLegend = plt.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2,  mode="expand", borderaxespad=0.,fontsize = 'x-small')
    sio = cStringIO.StringIO()
    myFIGA1.savefig(sio, dpi=300, format='png', bbox_extra_artists=(myLegend,), bbox_inches='tight') #myFIGA1.savefig("C:/Python27/interBLX1.png", dpi=300, format='png', bbox_extra_artists=(myLegend,), bbox_inches='tight')
    saveas = pickle.dumps(myFIGA1)  ### 
    fig1_encoded = base64.b64encode(saveas)
    #plt.tight_layout(rect = (0,0,0.5,1))
    plt.show()
    """

    #without subplot
    #myLegend = plt.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2,  mode="expand", borderaxespad=0.,fontsize = 'x-small')
    #myLegend = plt.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2, borderaxespad=0.,fontsize = 'x-small')

    plt.xlabel("Measurements")
    plt.ylabel("Number of Labs")
    plt.title('Histogram of data as reported')

    #with subplot
    ax  = myFIGA1a.add_subplot(111)
    #myLegend = ax.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2,  mode="expand", borderaxespad=0.,fontsize = 'x-small')
    myLegend = ax.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2, borderaxespad=0.,fontsize = 'x-small')

    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_xaxis().set_ticks([])

    ## String IO
    #sio = cStringIO.StringIO()
    #myFIGA1a.savefig(sio, dpi=300, format='png', bbox_extra_artists=(myLegend,), bbox_inches='tight')
    #saveas = pickle.dumps(sio.getvalue()) ###   
    #fig1a_encoded = base64.b64encode(saveas)

    plt.tight_layout()
    #plt.show() #HIDE show on production
    sio = BytesIO()
    myFIGA1a.savefig(sio, dpi=300, format='png', bbox_extra_artists=(myLegend,),bbox_inches='tight') 
    sio.seek(0)
    fig1a_encoded = base64.b64encode(sio.getvalue())

    plt.close()
	

    ######################## test decoder
    ## THIS WORKS WITH::: saveas = pickle.dumps(sio.getvalue()) ###
    #decc = base64.standard_b64decode(fig1_encoded) 
    #mystr = pickle.loads(decc)
    #stb = cStringIO.StringIO(mystr)
    #img = Image.open(stb)
    #img.seek(0)
    #plt.imshow(img)
    ## THIS WORKS ON SAVED IMAGE ############
    #stb = cStringIO.StringIO(sio.getvalue())
    #img = Image.open(stb)
    #img.seek(0)
    #plt.imshow(img)
    #########################end test decoder

	
    myFIGA1b = plt.figure()

    X = []
    Y = []
    for i in range (len(bb)-1):
        X.append( (bb[i] + bb[i+1]) / 2 )
        Y.append(0)
    #print X, "\n", Y
    plt.plot(Y,X, 'r--')
    for i in range (len(X)):
        plt.text(Y[i]+0.1, X[i], labels_copy[i])
    plt.axis([0, 1, min(bb), abs((min(bb)*0.05)+max(bb))]) ##abs

    plt.xlabel(header[0])
    plt.ylabel(header[1])

    #plt.xlabel("Laboratories")
    #plt.ylabel("Measurements")
    plt.title('Histogram of data as reported')

    ax  = myFIGA1b.add_subplot(111)
    #myLegend = ax.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2, borderaxespad=0.,fontsize = 'x-small')
    ax.axes.get_xaxis().set_ticks([])

    plt.tight_layout()
    #plt.show() #HIDE show on production
    sio = BytesIO()
    myFIGA1b.savefig(sio, dpi=300, format='png') 
    sio.seek(0)
    fig1b_encoded = base64.b64encode(sio.getvalue())

    ## String IO version
    #sio = cStringIO.StringIO()
    #myFIGA1b.savefig(sio, dpi=300, format='png') 
    #saveas = pickle.dumps(sio.getvalue())
    #fig1b_encoded = base64.b64encode(saveas)

    plt.close()


    return fig1a_encoded, fig1b_encoded

"""
     Create histograms for report (on raw data)
"""
def hist_bias(num_bins, my_list_transposed, header):

    bins = numpy.linspace(min(my_list_transposed[1]), max(my_list_transposed[1]), num_bins+1)

    labels = []
    for i in range (num_bins):
        labels.append("")

    for i in range (len(my_list_transposed[1])):
        for j in range (num_bins):
            if my_list_transposed[1][i] >= bins[j] and my_list_transposed[1][i] <= bins[j+1]:
                labels[j] = labels[j] + "    " + str(my_list_transposed[0][i])

    colours = []
    #print num_bins
    for i in range (num_bins):
        colours.append(random.rand(3,1)) # random colour scheme
                

    #myFIGA1 = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') ###
    myFIGA2a = plt.figure()

    # hist returns: num inst, bin bound, patches
    nn, bb, pp  = plt.hist(my_list_transposed[1], bins=num_bins, histtype='barstacked', color = random.rand(3,1), stacked=True) # , normed=True

    ## copy labels for 2nd histogram (first copy gets altered here)
    labels_copy = deepcopy(labels) 

    for i in range (len(pp)-1,0,-1):
        if labels[i] == "":
            labels.pop(i)
            pp.pop(i)
            colours.pop(i)
    for i in range (len(pp)):
        plt.setp(pp[i], color=colours[i])  


    myLegend = plt.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2, borderaxespad=0.,fontsize = 'x-small')

    plt.xlabel("Values")
    plt.ylabel("Number of Laboratories")
    plt.title('Histogram of estimates of laboratory bias')

    #ax  = myFIGA2a.add_subplot(111)
    #myLegend = ax.legend(pp, labels,bbox_to_anchor=(1.05, 1, 1.25, 0), loc=2, borderaxespad=0.,fontsize = 'x-small')
    #ax.axes.get_xaxis().set_ticks([])

    plt.tight_layout()
    #plt.show() #HIDE show on production
    sio = BytesIO()
    myFIGA2a.savefig(sio, dpi=300, format='png', bbox_extra_artists=(myLegend,),bbox_inches='tight') 
    sio.seek(0)
    fig2a_encoded = base64.b64encode(sio.getvalue())

    # SIO Version
    #sio = cStringIO.StringIO()
    #myFIGA2a.savefig(sio, dpi=300, format='png', bbox_extra_artists=(myLegend,), bbox_inches='tight')
    #saveas = pickle.dumps(sio.getvalue()) ###   
    #fig2a_encoded = base64.b64encode(saveas)

    plt.close()

	
    myFIGA2b = plt.figure()

    X = []
    Y = []
    for i in range (len(bb)-1):
        X.append( (bb[i] + bb[i+1]) / 2 )
        Y.append(0)
    plt.plot(Y,X, 'r--')
    for i in range (len(X)):
        plt.text(Y[i]+0.1, X[i], labels_copy[i])
    plt.axis([0, 1, min(bb), abs(min(bb))+max(bb)]) ##abs
    
    #plt.xlabel(header[0])
    #plt.ylabel(header[1])
    plt.xlabel(" Laboratories")
    plt.ylabel("Bias Estimate")
    plt.title('Histogram of estimates of laboratory bias')
    
    ax  = myFIGA2b.add_subplot(111)
    ax.axes.get_xaxis().set_ticks([])

    plt.tight_layout()
    #plt.show() #HIDE show on production
    sio = BytesIO()
    myFIGA2b.savefig(sio, dpi=300, format='png') 
    sio.seek(0)
    fig2b_encoded = base64.b64encode(sio.getvalue())

    #sio = cStringIO.StringIO()
    #myFIGA2b.savefig(sio, dpi=300, format='png') 
    #saveas = pickle.dumps(sio.getvalue())
    #fig2b_encoded = base64.b64encode(saveas)

    plt.close()


    return fig2a_encoded, fig2b_encoded

"""
    Plot Ranks and error bars
"""
def plot_ranks (ranks, ranks_pc, values, uncertainties, rob_avg, uncertaintyAV):
    myFIGA3 = plt.figure()
    
    #0502
    xx = [x for y, x in sorted(zip(ranks, values))]
    yy = [y for y, x in sorted(zip(ranks, values))]
    #print "ranks, values", ranks, values, "\n"
    #print xx,yy

    plt.plot(ranks, values, 'ro')
    #plt.plot(ranks, values)
    plt.errorbar(ranks, values, yerr = uncertainties, fmt='o')
    plt.xlim([round(min(ranks)) - 1, round(max(ranks)) + 1,])
    #sio = cStringIO.StringIO()
    #myFIGA3.savefig(sio, dpi=300, format='png') 
    #saveas = pickle.dumps(sio.getvalue())
    #fig3_encoded = base64.b64encode(saveas)
    exp_U_95_plus = round(rob_avg + 2*uncertaintyAV)
    exp_U_95_minus = round(rob_avg - 2*uncertaintyAV)
    plt.axhline(y=round(rob_avg),c="green",linewidth=1.5,zorder=0)
    plt.axhline(y=exp_U_95_plus,c="green",linewidth=0.5,zorder=0, ls = 'dashed')
    plt.axhline(y=exp_U_95_minus,c="green",linewidth=0.5,zorder=0, ls = 'dashed')

    plt.xlabel("Laboratory Ranks")
    plt.ylabel("Values and uncertainties")
    plt.title('Normal probability plot of expanded uncertainties for lab rankings')

    plt.tight_layout()
    #plt.show() #HIDE show on production
    sio = BytesIO()
    myFIGA3.savefig(sio, dpi=300, format='png') 
    sio.seek(0)
    fig3_encoded = base64.b64encode(sio.getvalue())

    plt.close()

    myFIGA4 = plt.figure()
    plt.plot(ranks_pc, values, 'ro')
    #plt.plot(ranks_pc, values)
    plt.errorbar(ranks_pc, values, yerr = uncertainties, fmt='o')
    plt.xlim([round(min(ranks_pc)) - 1, round(max(ranks_pc)) + 1,])
    #sio = cStringIO.StringIO()
    #myFIGA4.savefig(sio, dpi=300, format='png') 
    #saveas = pickle.dumps(sio.getvalue())
    #fig4_encoded = base64.b64encode(saveas)
    plt.axhline(y=round(rob_avg),c="green",linewidth=1.5,zorder=0)
    plt.axhline(y=exp_U_95_plus,c="green",linewidth=0.5,zorder=0, ls = 'dashed')
    plt.axhline(y=exp_U_95_minus,c="green",linewidth=0.5,zorder=0, ls = 'dashed')

    plt.xlabel("Laboratory % Ranks")
    plt.ylabel("Values and uncertainties")
    plt.title('Normal probability plot of expanded uncertainties for % ranking of labs')

    plt.tight_layout()
    #plt.show() #HIDE show on production
    sio = BytesIO()
    myFIGA4.savefig(sio, dpi=300, format='png') 
    sio.seek(0)
    fig4_encoded = base64.b64encode(sio.getvalue())

    plt.close()
    return fig3_encoded, fig4_encoded

"""
    plot ranks % against raw values 
"""
def ranks_vs_values (values, ranks_pc, z_scores):
    myFIGA5 = plt.figure()
    z = numpy.polyfit(ranks_pc,values,1)
    p = numpy.poly1d(z)

    plt.plot(ranks_pc, values, 'ro', ranks_pc, p(ranks_pc), 'r--')
    #plt.plot(ranks_pc, values, 'ro')
    plt.xlim([round(min(ranks_pc)) - 1, round(max(ranks_pc)) + 1,])

    # loop for outliers AFTER Z SCORE TEST ###
    for i in range (len(ranks_pc)):
        if z_scores [i] < -2 or z_scores [i] > 2:
            #plt.annotate("Z = " + str(z_scores[i]), xy = (ranks_pc[i], values[i]), xytext=(round(ranks_pc[i] + 0.05*max(ranks_pc)), round(values[i] + 0.05*max(values))), arrowprops=dict(facecolor='black', shrink=0.05))
            plt.annotate("Z = " + str(z_scores[i]), xy = (ranks_pc[i], values[i]), xytext=(round(ranks_pc[i] - 0.05*max(ranks_pc)), round(values[i] - 0.05*max(values))), arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel("Laboratory Percentage Rank")
    plt.ylabel("Values")
    plt.title('Normal probability plot of measurement results')
    #ax  = myFIGA5.add_subplot(111)
    #ax.axes.get_xaxis().set_ticks([])

    plt.tight_layout()
    #plt.show() #HIDE show on production
    sio = BytesIO()
    myFIGA5.savefig(sio, dpi=300, format='png', bbox_inches="tight") 
    sio.seek(0)
    fig5_encoded = base64.b64encode(sio.getvalue())

    #sio = cStringIO.StringIO()
    #myFIGA5.savefig(sio, dpi=300, format='png') 
    #saveas = pickle.dumps(sio.getvalue())
    #fig5_encoded = base64.b64encode(saveas)


    plt.close()
    return fig5_encoded

"""
    Conceal Lab names 
"""
def anonimise_labs (myList):
    nameDic = {}
    for i in range (len(myList[0])):
        nameDic["Row_" + str(i+1)] = ["Lab_" + str(i+1),  myList[0][i]]
        myList[0][i] = "Lab_" + str(i+1)

    return myList, nameDic

"""
    Ranks and percentage ranks
"""
def get_ranks (data_list):
    rank = []
    i,j = 0,0
    while 1:
        if i < len(data_list):
            count_rank = 1
            count_replicate = 1
            j = 0
            while 2 and j < len(data_list):
                if (data_list[i] > data_list[j]) :
                    count_rank += 1
                if (data_list[i] == data_list[j] and i !=j) :
                    count_replicate += 1
                j += 1
            if count_replicate >1:
                count_repl_rank = 0
                for repl in range (count_replicate):
                    count_repl_rank = count_repl_rank + count_rank + repl
                rank.append( count_repl_rank / count_replicate)
            else:
                rank.append(count_rank) 
            i += 1
        else:
            break
        
    rank_pc = []
    for i in range (len(rank)):
        #rank_pc.append(round(100*(rank[i] - 0.5)/ len(rank),2)) # 2 decimals
        rank_pc.append(round(100*(rank[i] - 0.5)/ len(rank),0))
    #print "rank, datalist", rank, data_list, "\n"
    return rank, rank_pc

"""
    Ranks and percentage ranks
def get_ranks (data_list):

    rank = []

    i,j = 0,1
    while 1:
        if i < len(data_list):
            count_rank = 0
            j = i + 1
            while 2 and j < len(data_list):
                if (data_list[i] == data_list[j]) :
                    count_rank += 1
                    j += 1
                else:
                    break
            
            if count_rank == 0:
                rank.append(i+1)
                i += 1
            else:
                avg_rank = float(2*i+1 + count_rank) / 2.0 #####
                for k in range (count_rank):
                    rank.append(avg_rank)
                i += count_rank
        else:
            break

    rank_pc = []
    for i in range (len(rank)):
        #rank_pc.append(round(100*(rank[i] - 0.5)/ len(rank),2)) # 2 decimals
        rank_pc.append(round(100*(rank[i] - 0.5)/ len(rank),0))
    print "rank, datalist", rank, data_list, "\n"
    return rank, rank_pc
"""

"""
    Get ranks and percentage ranks dictionary for report
"""
def get_rank_dict(labs,rank, rank_pc, titlesAsRows = False):
    myDict = {}
    if titlesAsRows == True:
        for i in range (len (labs)):
            myDict[labs[i]] = [rank[i], rank_pc[i]]
    else: 
        for i in range (len (labs)):
            myDict["Row_" + str(i+1)] = [labs[i], rank[i], rank_pc[i]]
    return myDict

"""
    Rob Avg, Rob St Dev, U_AV, raw data histograms
"""
def interlab_test_pt1 (headers, myData, myDataTransposed ):

    # get data
    #headers, myData, myDataTransposed = get_data() # transposed == REVERSED INDICES
    #print "\n\n\n", myData, myDataTransposed , "\n\n\n"
    dataTransposed = deepcopy(myDataTransposed) ### data may be modified depending on extreme values - use for plots
    
    #print "\nHeaders: ", headers
    #print "\nSorted Data: ", myData
    #print "\nProcessed Data: ", myDataTransposed

    data_list, robust_avg_x, robust_std_s = loop_algorithm_a(myDataTransposed[1]) # 'values' index
    #print robust_avg_x, robust_std_s

    # if uncertainties have been reported:
    if uncertainties_exist(myDataTransposed[2]) == 1:
        tempU = get_uncertainty_for_assigned_value(robust_std_s, len(myDataTransposed[0]), uncertainties = myDataTransposed[2]) # 'uncertainty' index
    else:
        tempU = get_uncertainty_for_assigned_value(robust_std_s, len(myDataTransposed[0]), uncertainties = [])
    #print "\nUncertainty = ", tempU

    #dataTransposed = kill_outliers(dataTransposed, robust_avg_x, robust_std_s) ## should we kill outliers?

    # Optimal No. bins according to Scott's rule
    test_bin = int ( (3.5*robust_std_s) / numpy.power(len(dataTransposed[1]), 0.3))
    if test_bin >20:
        test_bin = 20
    #print "\nCalculated bins: ", test_bin
    fig1a, fig1b = hist_plots(test_bin, dataTransposed, headers)

    """
        lab_bias = get_diff (myDataTransposed[0], tempX)
        print "\nLab Bias: ", myDataTransposed[0], lab_bias

        # lab_bias = x-X, tempS = rob std dev
        signals = check_bias(lab_bias, tempS) 
        # list of signals, lab names, lab bias
        print_labs_with_signals (signals, myDataTransposed[1], lab_bias) 

        lab_bias_percent = get_diff_percent (myDataTransposed[0], tempX)
        print "\nLab Bias %%: ", lab_bias_percent

        signals_percent = check_bias(lab_bias_percent, tempS) 
        print_labs_with_signals (signals, myDataTransposed[1], lab_bias_percent) 

        test_bin = 18 # normalise somehow
        #test_bin = int ( (3.5*tempS) / numpy.power(len(dataTransposed[1]), 0.3))

        # construct matrix with labels and values first
        percent_for_hist = []
        percent_for_hist.append(myDataTransposed[1]) 
        percent_for_hist.append(lab_bias_percent)
        hist_plots(test_bin, percent_for_hist, headers)

        # return ranks
        rank, rank_pc = get_ranks(myDataTransposed[1])
        print "\nRanks: ", rank
        print "\nRanks %%: ", rank_pc
    """
    return robust_avg_x, robust_std_s, tempU, fig1a, fig1b

"""
    Check if uncertainties exist
"""
def uncertainties_exist(u_list):
    flag_U = 0
    for i in range (len(u_list)):
        if u_list[i] !=0:
            flag_U = 1
            break
    return flag_U

"""
    Replace uncertainties with u_AV if uncertainties do not exist
"""
def replace_uncertainties(u_list, u_AV):
    for i in range (len(u_list)):
        u_list[i] = u_AV
    return u_list


"""
    Get Diff & Diff %
"""
def get_diffs(data, ass_value): 
    #print data, ass_value
    diff = []
    diff_pc = []
    for i in range (len(data)): 
        if ass_value == 0:
            diff.append(round(data[i],2))
            diff_pc.append(round(100*data[i],2))
        else:
            #diff.append(round((data[i] - ass_value)/ass_value,2))
            diff.append(round((data[i] - ass_value),2))
            diff_pc.append(round(100*(data[i] - ass_value)/ass_value,2))
    return diff, diff_pc

"""
    z-scores
"""
def z_scores (list, avg, std):
    z_scorez = []
    for i in range (len(list)):
        z_scorez.append( round((float(list[i]) - avg )/ std, 2))
    return z_scorez

"""
    e-values
"""
def e_value (list, rob_avg, rob_std, u_AV, u_lab):
    Uref = u_AV # should be matrix?

    Eval95 = []
    Eval99 = []
    for i in range (len(list)):
        Eval95.append(round((list[i] - rob_avg) / numpy.sqrt( numpy.power(2*u_lab[i],2) + numpy.power(2*Uref,2)),2))
        Eval99.append(round((list[i] - rob_avg) / numpy.sqrt( numpy.power(3*u_lab[i],2) + numpy.power(3*Uref,2)),2))
    return Eval95, Eval99

"""
    zeta-scores
"""
def zeta_scores (list, rob_avg, rob_std, u_AV, u_lab):
    Uref = u_AV # should be matrix?

    zeta = []
    for i in range (len(list)):
        zeta.append(round((list[i] - rob_avg) / numpy.sqrt( numpy.power(u_lab[i],2) + numpy.power(Uref,2)),2))
    return zeta

"""
    z'
"""
def z_tonos (list, rob_avg, rob_std, u_AV):
    z_ton = []
    Ulab = u_AV # check uncertainty
    for i in range (len(list)):
        z_ton.append(round((list[i] - rob_avg) / numpy.sqrt( numpy.power(Ulab,2) + numpy.power(rob_std,2)),2))
    return z_ton

"""
    Ez scores
"""
def ez_scores(list, rob_avg, rob_std, u_AV, u_lab):
    Uref = u_AV # should be matrix?

    Ez_minus95 = []
    Ez_plus95 = []
    Ez_minus99 = []
    Ez_plus99 = []

    temp_U_override = 0.001
    for i in range (len(list)):
        if u_lab[i] !=0:
            Ez_minus95.append(round((list[i] - (rob_avg - (2*u_AV)))/ (2*u_lab[i]),2))
            Ez_plus95.append(round((list[i] - (rob_avg - (2*u_AV)))/ (2*u_lab[i]),2)) 
            Ez_minus99.append(round((list[i] - (rob_avg - (3*u_AV)))/ (3*u_lab[i]),2))
            Ez_plus99.append(round((list[i] - (rob_avg - (3*u_AV)))/ (3*u_lab[i]),2)) 
            ###print "!=0", list[i], rob_avg, u_AV, u_lab[i],(list[i] - (rob_avg - (2*u_AV)))/ (2*u_lab[i])
        elif u_AV !=0:
            Ez_minus95.append(round((list[i] - (rob_avg - (2*u_AV)))/ (2*u_AV),2)) #new 06042015
            Ez_plus95.append(round((list[i] - (rob_avg - (2*u_AV)))/ (2*u_AV),2)) 
            Ez_minus99.append(round((list[i] - (rob_avg - (3*u_AV)))/ (3*u_AV),2))
            Ez_plus99.append(round((list[i] - (rob_avg - (3*u_AV)))/ (3*u_AV),2))        
        else:
            #Ez_minus95.append((list[i] - (rob_avg - (2*u_AV)))/ (2*temp_U_override))
            #Ez_plus95.append((list[i] - (rob_avg - (2*u_AV)))/ (2*temp_U_override)) 
            #Ez_minus99.append((list[i] - (rob_avg - (3*u_AV)))/ (3*temp_U_override))
            #Ez_plus99.append((list[i] - (rob_avg - (3*u_AV)))/ (3*temp_U_override)) 
            Ez_minus95.append(-3)
            Ez_plus95.append(3) 
            Ez_minus99.append(-3)
            Ez_plus99.append(3) 
            ###print "==0", list[i], rob_avg, u_AV, u_lab[i],(list[i] - (rob_avg - (2*u_AV)))/ (2*temp_U_override)
    return Ez_minus95, Ez_plus95, Ez_minus99, Ez_plus99


"""
    z-scores, e-values, etc.
"""
def interlab_test_pt2 (full_list, rob_avg, rob_std, u_AV):
    labz = full_list[0]
    valz = full_list[1]
    u_lab = full_list[2]
    z_sc = z_scores(valz, rob_avg, rob_std) 
    e_val95, e_val99 = e_value (valz, rob_avg, rob_std, u_AV, u_lab)
    z_ton = z_tonos (valz, rob_avg, rob_std, u_AV)   
    zeta = zeta_scores (valz, rob_avg, rob_std, u_AV, u_lab)
    diff, diff_pc = get_diffs(valz, rob_avg)
    Ez_minus95, Ez_plus95, Ez_minus99, Ez_plus99 = ez_scores (valz, rob_avg, rob_std, u_AV, u_lab)

    dict_z_sc = {}
    dict_e_val95 = {}
    dict_e_val99 = {}
    dict_z_ton = {}
    dict_zeta = {}
    dict_ez_minus95 = {}
    dict_ez_plus95 = {}
    dict_ez_minus99 = {}
    dict_ez_plus99 = {}
	
    for i in range (len(labz)):
        dict_z_sc[labz[i]] = z_sc[i]
        dict_e_val95[labz[i]] = e_val95[i]
        dict_e_val99[labz[i]] = e_val99[i]
        dict_z_ton[labz[i]] = z_ton[i]
        dict_zeta[labz[i]] = zeta[i]
        dict_ez_minus95[labz[i]] = Ez_minus95[i]
        dict_ez_plus95[labz[i]] = Ez_plus95[i]
        dict_ez_minus99[labz[i]] = Ez_minus99[i]
        dict_ez_plus99[labz[i]] = Ez_plus99[i]

    return labz, z_sc, e_val95, e_val99, z_ton, zeta, Ez_minus95, Ez_plus95, Ez_minus99, Ez_plus99, dict_z_sc, dict_e_val95, dict_e_val99, dict_z_ton, dict_zeta, dict_ez_minus95, dict_ez_plus95, dict_ez_minus99, dict_ez_plus99

"""
    individual z-scores
"""
def individual_z_scores (labz, z_scores):
    namez = []
    if isinstance(z_scores[0], list):
        print "Currently do not test on multiple outcomes"
        """
        for i in range len(z_scores):
            namez.append([])
        position_diff = 1/len(z_scores)
        for i in range len(z_scores):
            namez[i].append(...) ##
        for i in range len(z_scores):
            plt.bar( namez[i], z_scorez[i], width = 0.1, color='...') ##
        """
        fig6_encoded = base64.b64encode("")
    else:
        for i in range (len(z_scores)): # labs
            namez.append(i+1)
        
        myFIGA6 = plt.figure()
        plt.bar( namez, z_scores, width = 0.1, color='b')
        tix = ["0"] + labz
        plt.xticks(range(len(tix)),tix)
        plt.xlabel("Laboratory")
        plt.ylabel("z-scores")

        plt.title('z-scores comparison per lab')
        #ax  = myFIGA6.add_subplot(111)
        #ax.axes.get_xaxis().set_ticks([])

        plt.tight_layout()
        #plt.show() #HIDE show on production
        sio = BytesIO()
        myFIGA6.savefig(sio, dpi=300, format='png', bbox_inches="tight") 
        sio.seek(0)
        fig6_encoded = base64.b64encode(sio.getvalue())

        #sio = cStringIO.StringIO()
        #myFIGA6.savefig(sio, dpi=300, format='png') 
        #saveas = pickle.dumps(sio.getvalue())
        #fig6_encoded = base64.b64encode(saveas)

        plt.close()
        
    plt.close()
    return fig6_encoded
    

"""
    Egg plot coordinates
"""
def get_sigmas_exes(robust_average, robust_std_dev, chi_square, number_replicate_experiments):
    #print robust_average, robust_std_dev, chi_square, number_replicate_experiments
    x_min = robust_average - (robust_std_dev * numpy.sqrt(chi_square / number_replicate_experiments ))
    x_max = robust_average + (robust_std_dev * numpy.sqrt(chi_square / number_replicate_experiments ))

    y_neg = []
    y_pos = []
    xstep = (x_max-x_min)/1000 ## and below
    xx  = x_min
    x_mat = []

    for i in range (1,1000):
        y_neg.append(robust_std_dev * numpy.exp( (-1.0/(numpy.sqrt(2*(number_replicate_experiments - 1)))) * \
                 numpy.sqrt(abs(chi_square - numpy.power(numpy.sqrt(number_replicate_experiments) * ((xx - robust_average)/robust_std_dev),2)))))
        y_pos.append(robust_std_dev * numpy.exp( (1.0/(numpy.sqrt(2*(number_replicate_experiments - 1)))) * \
                 numpy.sqrt(abs(chi_square - numpy.power(numpy.sqrt(number_replicate_experiments) * ((xx - robust_average)/robust_std_dev),2)))))

        xx += xstep
        x_mat.append(xx)

    # evaluate x
    #print "\n\nEVAL", max(x_mat), "=?=", max(y_pos)
    x_mat.append(x_max)

    # evaluate y
    y_neg.append(robust_std_dev * numpy.exp( (-1.0/(numpy.sqrt(2*(number_replicate_experiments - 1)))) * \
                 numpy.sqrt(abs(chi_square - numpy.power(numpy.sqrt(number_replicate_experiments) * ((x_max - robust_average)/robust_std_dev),2))))) #abs 
    y_pos.append(robust_std_dev * numpy.exp( (1.0/(numpy.sqrt(2*(number_replicate_experiments - 1)))) * \
                 numpy.sqrt(abs(chi_square - numpy.power(numpy.sqrt(number_replicate_experiments) * ((x_max - robust_average)/robust_std_dev),2))))) #abs
    y_neg_mat = deepcopy(y_neg)
    y_pos_mat = deepcopy(y_pos)
    #print y_neg_mat[len(y_neg_mat)-1], "=?=", y_pos_mat[len(y_pos_mat)-1], "len = ", len(y_neg_mat)

    return x_mat, y_neg_mat, y_pos_mat

"""
    Algorithm S
"""
def algorithm_s(w, w_star, instances, replicates):
    dof_t =[[0, 0], [1.645, 1.097], [1.517, 1.054], [1.444, 1.039], [1.395, 1.032], [1.359, 1.027], 
            [1.332, 1.024], [1.310, 1.021], [1.292, 1.019], [1.277, 1.018], [1.264, 1.017]]

    if instances == "value":
        dof = 1
    elif instances == "std":
        if replicates -1 >10:
            dof = 10 
        else:
            dof = replicates -1
    else:
        dof = 0 #initial
    #print "Degrees of F = ", dof
    # d-o-f_table [d-o-f][0:heta, 1:ksi]
    psi = dof_t[dof][0] * w_star # heta * w_star
    #print psi

    w_new = deepcopy(w)
    for i in range(len(w)):
        if w[i] > psi: 
            w_new[i] = psi
        else:
            w_new[i] = w[i]
    #print w_new

    w_star_new = dof_t[dof][1] * numpy.sqrt( sum( numpy.power(w_new, 2) ) / len(w_new) )

    converged = 0

    #if w_star == w_star_new:
    if w == w_new:
        converged = 1
    #print converged, w_star, w_star_new
        
    return w_new, w_star_new, converged

"""
    Loops Algorithm S until it converges
"""
def loop_algorithm_s (w, myType, replicates): # myType == "value" OR "std"
    w_star = numpy.median(w)
    while 1:
        w, w_star, c = algorithm_s(w, w_star, myType, replicates)
        if c == 0:
            #print w
            #print w_star
            continue
        else:
            #print "converged !!"
            break
    #print w, w_star
    return w, w_star, c 

"""
    Testing function - make sure to delete
"""
def delete_this():
    datalist = [[2.15, 0.13],[1.85, 0.21],[1.80, 0.08],[1.80, 0.24],[1.90, 0.36],[1.90, 0.32],[1.90, 0.14],[2.05, 0.26],[2.35, 0.39], \
    [2.03, 0.53],[2.08, 0.25],[1.25, 0.24],[1.13, 0.72],[1.00, 0.26],[1.08, 0.17],[1.20, 0.32],[1.35, 0.4],[1.23, 0.36], \
    [1.23, 0.33],[0.90, 0.43],[1.48, 0.40],[1.20, 0.55],[1.73, 0.39],[1.43, 0.30],[1.28, 0.22]]
    averages=[]
    std_devs=[]
    for i in range (len(datalist)):
        averages.append(datalist[i][0])
        std_devs.append(datalist[i][1])
    robust_average = 1.57
    robust_std_dev = 0.34
    return datalist, averages, std_devs, robust_average, robust_std_dev


"""
    Std Devs v Averages plots
"""
def std_v_avg (datalist, robust_average):
    dof = [5.99, 9.21, 13.82]
    
    averages = []
    std_devs = []
    for i in range (len(datalist)): 
        averages.append(numpy.mean(datalist[i]))
        std_devs.append(numpy.std(datalist[i]))
    #print averages
                                                                # list, type, replicates
    new_std_devs, robust_std_dev, converged = loop_algorithm_s (std_devs, "std", len(datalist[0]))
    #print "ROB STD = ", robust_std_dev
    #print "New list: ", new_std_devs

    x_mat0, y_neg_mat0, y_pos_mat0 = get_sigmas_exes(robust_average, robust_std_dev, dof[0], number_replicate_experiments = len(datalist[0])) 
    x_mat1, y_neg_mat1, y_pos_mat1 = get_sigmas_exes(robust_average, robust_std_dev, dof[1], number_replicate_experiments = len(datalist[0]))
    x_mat2, y_neg_mat2, y_pos_mat2 = get_sigmas_exes(robust_average, robust_std_dev, dof[2], number_replicate_experiments = len(datalist[0]))

    myFIGA7 = plt.figure()
    plt.plot(averages, std_devs, 'ro') 
    plt.plot(x_mat0,y_neg_mat0)
    plt.plot(x_mat0,y_pos_mat0)
    plt.plot(x_mat1,y_neg_mat1)
    plt.plot(x_mat1,y_pos_mat1)
    plt.plot(x_mat2,y_neg_mat2)
    plt.plot(x_mat2,y_pos_mat2)

    #plt.annotate("5% level", xy = (max(x_mat0), numpy.mean(y_pos_mat0)), xytext=(round(max(x_mat0) + 0.05*max(x_mat0)), round(numpy.mean(y_pos_mat0) + 0.05*numpy.mean(y_pos_mat0))), arrowprops=dict(facecolor='black', shrink=0.05))
    #plt.annotate("1% level", xy = (max(x_mat1), numpy.mean(y_pos_mat1)), xytext=(round(max(x_mat1) + 0.05*max(x_mat1)), round(numpy.mean(y_pos_mat1) + 0.05*numpy.mean(y_pos_mat1))), arrowprops=dict(facecolor='black', shrink=0.05))
    #plt.annotate("0.1% level", xy = (max(x_mat2), numpy.mean(y_pos_mat2)), xytext=(round(max(x_mat2) + 0.05*max(x_mat2)), round(numpy.mean(y_pos_mat2) + 0.05*numpy.mean(y_pos_mat2))), arrowprops=dict(facecolor='black', shrink=0.05))

    ax  = myFIGA7.add_subplot(111)
    #print "\n\n\n", numpy.mean(x_mat0), numpy.mean(y_neg_mat0), numpy.mean(y_pos_mat0), "\n\n\n"
    ax.annotate("5% level", xy = (numpy.mean(x_mat0), max(y_pos_mat0)), xytext=(round(numpy.mean(x_mat0) + 0.05*numpy.mean(x_mat0)), round(max(y_pos_mat0) + 0.05*max(y_pos_mat0))), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate("1% level", xy = (numpy.mean(x_mat1), max(y_pos_mat1)), xytext=(round(numpy.mean(x_mat1) + 0.05*numpy.mean(x_mat1)), round(max(y_pos_mat1) + 0.05*max(y_pos_mat1))), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate("0.1% level", xy = (numpy.mean(x_mat2), max(y_pos_mat2)), xytext=(round(numpy.mean(x_mat2) + 0.05*numpy.mean(x_mat2)), round(max(y_pos_mat2) + 0.05*max(y_pos_mat2))), arrowprops=dict(facecolor='black', shrink=0.05))


    #plt.show() # hide on production
    plt.xlabel("Average")
    plt.ylabel("Standard Deviation")

    plt.title('Plot of Standard Deviations against Averages including 0.1%, 1% and 5% levels')

    plt.tight_layout()
    #plt.show() #HIDE show on production
    sio = BytesIO()
    myFIGA7.savefig(sio, dpi=300, format='png', bbox_inches="tight") 
    sio.seek(0)
    fig7_encoded = base64.b64encode(sio.getvalue())
    #sio = cStringIO.StringIO()
    #myFIGA7.savefig(sio, dpi=300, format='png') 
    #saveas = pickle.dumps(sio.getvalue())
    #fig7_encoded = base64.b64encode(saveas)
    plt.close()
    return fig7_encoded


"""
    Plot bias histograms
"""
def plot_bias(labz, diff, robust_avg_x, robust_std_s):
    dataTransposed = []
    dataTransposed.append(labz)
    dataTransposed.append(diff)
    test_bin = int ( (3.5*robust_std_s) / numpy.power(len(dataTransposed[1]), 0.3))
    if test_bin >20:
        test_bin = 20
    headers=["labs", "bias"]
    fig2a, fig2b = hist_bias(test_bin, dataTransposed, headers)
    return fig2a, fig2b

"""
    Check lab stats for warning/action signals
"""
def check_stats (robAA, robSS, diff, diff_pc, stat_matrix):
    #stat_matrix = [labz, z_sc, e_val95, e_val99, z_ton, zeta, Ez_minus95, Ez_plus95, Ez_minus99, Ez_plus99]
    labz = stat_matrix[0]
    z_sc = stat_matrix[1]
    e_val95 = stat_matrix[2]
    e_val99 = stat_matrix[3]
    z_ton = stat_matrix[4]
    zeta = stat_matrix[5]
    Ez_minus95 = stat_matrix[6]
    Ez_plus95 = stat_matrix[7]
    Ez_minus99 = stat_matrix[8]
    Ez_plus99 = stat_matrix[9]
    
    warning = []
    action = []

    for i in range (len(diff)):
        # diff
        if diff[i] > 3*robSS or diff[i] < -3*robSS:
            #action.append([labz[i], diff[i], "Action Based on Bias (Diff)"])
            action.append([labz[i], str(diff[i]), "Action Based on Bias (Diff)"])
        elif diff[i] > 2*robSS or diff[i] < -2*robSS:
            #warning.append([labz[i], diff[i], "Warning Based on Bias (Diff)"])
            warning.append([labz[i], str(diff[i]), "Warning Based on Bias (Diff)"])

        # diff%
        if diff_pc[i] > 300*robSS/robAA or diff_pc[i] < -300*robSS/robAA:
            #action.append([labz[i], diff_pc[i], "Action Based on Bias (Diff %)"])
            action.append([labz[i], str(diff_pc[i]), "Action Based on Bias (Diff %)"])
        elif diff_pc[i] > 200*robSS/robAA or diff_pc[i] < -200*robSS/robAA:
            #warning.append([labz[i], diff_pc[i], "Warning Based on Bias (Diff %)"])
            warning.append([labz[i], str(diff_pc[i]), "Warning Based on Bias (Diff %)"])

        # z-score
        if z_sc[i] > 3 or z_sc[i] < -3:
            #action.append([labz[i], z_sc[i], "Action Based on z-score"])
            action.append([labz[i], str(z_sc[i]), "Action Based on z-score"])
        elif z_sc[i] > 2 or z_sc[i] < -2:
            #warning.append([labz[i], z_sc[i], "Warning Based on z-score"])
            warning.append([labz[i], str(z_sc[i]), "Warning Based on z-score"])

        # en value 95% and 99%
        if abs(e_val95[i]) >=1:
            #warning.append([labz[i], e_val95[i], "Warning Based on En value (95% confidence level)"])
            warning.append([labz[i], str(e_val95[i]), "Warning Based on En value (95% confidence level)"])
        if abs(e_val95[i]) >=1:
            #warning.append([labz[i], e_val99[i], "Warning Based on En value (99% confidence level)"])
            warning.append([labz[i], str(e_val99[i]), "Warning Based on En value (99% confidence level)"])

        # z' score
        if z_ton[i] > 3 or z_ton[i] < -3:
            #action.append([labz[i], z_ton[i], "Action Based on z' score"])
            action.append([labz[i], str(z_ton[i]), "Action Based on z' score"])
        elif z_ton[i] > 2 or z_ton[i] < -2:
            #warning.append([labz[i], z_ton[i], "Warning Based on z' score"])
            warning.append([labz[i], str(z_ton[i]), "Warning Based on z' score"])

        # zeta score
        if zeta[i] > 3 or zeta[i] < -3:
            #action.append([labz[i], zeta[i], "Action Based on zeta score"])
            action.append([labz[i], str(zeta[i]), "Action Based on zeta score"])
        elif zeta[i] > 2 or zeta[i] < -2:
            #warning.append([labz[i], zeta[i], "Warning Based on zeta score"])
            warning.append([labz[i], str(zeta[i]), "Warning Based on zeta score"])

        if Ez_plus95[i] > 1 and Ez_minus95[i] <-1:
            #action.append([labz[i], [Ez_plus95[i], Ez_minus95[i]], "Action Based on Ez score (95% confidence level)"])
            action.append([labz[i], str([Ez_plus95[i], Ez_minus95[i]]), "Action Based on Ez score (95% confidence level)"])
        elif Ez_plus95[i] > 1 and Ez_minus95[i] >=-1:
            #warning.append([labz[i], Ez_plus95[i], "Warning Based on Ez+ score (95% confidence level)"])
            warning.append([labz[i], str(Ez_plus95[i]), "Warning Based on Ez+ score (95% confidence level)"])
        elif Ez_plus95[i] <= 1 and Ez_minus95[i] <-1:
            #warning.append([labz[i], Ez_minus95[i], "Warning Based on Ez- score (95% confidence level)"])
            warning.append([labz[i], str(Ez_minus95[i]), "Warning Based on Ez- score (95% confidence level)"])

        if Ez_plus99[i] > 1 and Ez_minus99[i] <-1:
            #action.append([labz[i], [Ez_plus99[i], Ez_minus99[i]], "Action Based on Ez score (99% confidence level)"])
            action.append([labz[i], str([Ez_plus99[i], Ez_minus99[i]]), "Action Based on Ez score (99% confidence level)"])
        elif Ez_plus99[i] > 1 and Ez_minus99[i] >=-1:
            #warning.append([labz[i], Ez_plus99[i], "Warning Based on Ez+ score (99% confidence level)"])
            warning.append([labz[i], str(Ez_plus99[i]), "Warning Based on Ez+ score (99% confidence level)"])
        elif Ez_plus99[i] <= 1 and Ez_minus99[i] <-1:
            #warning.append([labz[i], Ez_minus99[i], "Warning Based on Ez- score (99% confidence level)"])
            warning.append([labz[i], str(Ez_minus99[i]), "Warning Based on Ez- score (99% confidence level)"])

    return warning, action

"""
    Stats to dictionary
"""
def stats_to_dic (myList):
    statsDic = {}
    for i in range (len(myList)):
        statsDic["Row_" + str(i+1)] = myList[i]
    return statsDic

"""
    Make suggestions to labs
"""
def make_suggestions (labz, warning, action):
    action_sign = "Laboratory needs to proceed to corrective actions. These may include: \
                  \n a) checking that staff understand and follow the measurement procedure \
                  \n b) checking that all details of the measurement procedure are correct \
                  \n c) checking the calibration of equipment and the composition of reagents \
                  \n d) replacing suspect equipment or reagents \
                  \n e) comparative tests of staff, equipment and/or reagents with another laboratory"

    warn_sign = "Laboratory Performance is unsatisfactory and a warning must be issued."
    suggestions_dic = {}
    warning = map(list, zip(*warning)) 
    action = map(list, zip(*action)) 

    for i in range (len(labz)):
        if labz[i] in action[0]:
            suggestions_dic["Row_" + str(i+1)] = [labz[i], action_sign]
        elif labz[i] in warning[0]:
            suggestions_dic["Row_" + str(i+1)] = [labz[i], warn_sign]

    return suggestions_dic

###############################################################################################
# RUN AREA

@app.route('/pws/interlabtest', methods = ['POST'])
def create_task_interlabtest():

    if not request.json:
        abort(400)

    # data_list_transposed, data_list, data_with_replicates_transposed, data_with_replicates
    per_lab_data, per_attribute_data, per_lab_with_replicates, per_attribute_with_replicates = getJsonContents(request.json)
    #print per_lab_with_replicates, 
    #print per_attribute_with_replicates # check replicates

    per_attribute_data, name_dictionary = anonimise_labs(per_attribute_data)

    headers = "Labs", "Values", "Uncertainties"
    robAA, robSS, uncertaintyAV, fig1a, fig1b = interlab_test_pt1 (headers, per_lab_data, per_attribute_data)

    rank, rank_pc = get_ranks(per_attribute_data[1])

    # get_rank_dict(labs, ranks, ranks %)
    #rankDic = get_rank_dict(per_attribute_data[0],rank, rank_pc, titlesAsRows = True)
    rankDic = get_rank_dict(per_attribute_data[0],rank, rank_pc, titlesAsRows = False)

    # check uncertainties exist, otherwise use U of assigned value
    if uncertainties_exist(per_attribute_data[2]) == 0: 
        per_attribute_data[2] = replace_uncertainties(per_attribute_data[2], uncertaintyAV)

    # interlab_pt2 - stats
    labz, z_sc, e_val95, e_val99, z_ton, zeta, Ez_minus95, Ez_plus95, Ez_minus99, Ez_plus99, dict_z_sc, dict_e_val95, dict_e_val99, dict_z_ton, dict_zeta, dict_ez_minus95, dict_ez_plus95, dict_ez_minus99, dict_ez_plus99 = interlab_test_pt2 (per_attribute_data, robAA, robSS, uncertaintyAV)
    stat_matrix = [labz, z_sc, e_val95, e_val99, z_ton, zeta, Ez_minus95, Ez_plus95, Ez_minus99, Ez_plus99]

    stats_transposed = map(list, zip(*stat_matrix)) 
    stats_dic = stats_to_dic(stats_transposed)
 
    #warnings, actions = check_stats (stat_matrix)
    diff, diff_pc = get_diffs(per_attribute_data[1], robAA)

    fig2a, fig2b = plot_bias(labz, diff_pc, robAA, robSS) # return fig

    # Cumulative W&A signals
    warning, action = check_stats (robAA, robSS, diff, diff_pc, stat_matrix)
    warn_dic = stats_to_dic(warning)
    act_dic = stats_to_dic(action)
    #for i in range (len(warning)):
    #    print warning[i] 
    #for i in range (len(action)):
    #    print action[i] 

    suggestions_dic = make_suggestions (labz, warning, action)
    #print suggestions_dic

    # plot_ranks ( ranks , ranks % , values , uncertainties, u_AV)
    fig3,fig4 = plot_ranks (rank, rank_pc, per_attribute_data[1], per_attribute_data[2], robAA, uncertaintyAV)

    # plot ranks vs values
    fig5 = ranks_vs_values (per_attribute_data[1], rank_pc, z_sc)

    # bar plot z-scores
    fig6 = individual_z_scores (labz, z_sc) 

    fig7 = std_v_avg (per_attribute_with_replicates[1], robAA) ###

    task = {
        "singleCalculations": {"Robust Average": robAA, 
                               "Robust StDev": robSS,
                               "Uncertainty of Assigned Value": uncertaintyAV 
                              },
        "arrayCalculations": {"Ranking Array":
                               {"colNames": ["Rank", "Rank %"],
                                "values": rankDic
                               },
                              "Lab Real Names":
                               {"colNames": ["Given Name", "Original Name %"],
                                "values": name_dictionary
                               },
                              "Detailed Warning Signals":
                               {"colNames": ["Lab", "Problematic Values", "Signal Raised"],
                                "values": warn_dic
                               },
                              "Detailed Action Signals":
                               {"colNames": ["Lab", "Problematic Values", "Signal Raised"],
                                "values": act_dic
                               },
                              "Detailed Statistics":
                               {"colNames": ["Lab", "z-score", "En value (95%)", "En value (99%)", "z' score", "zeta score", "Ez- score (95%)", "Ez+ score (95%)", "Ez- score (99%)", "Ez+ score (99%)"],
                                "values": stats_dic
                               },
                              "Suggestions":
                               {"colNames": ["Lab", "Text"],
                                "values": suggestions_dic
                               }
                             },
        "figures": {"Figure 1: Colour Histogram for Lab Raw Data" : fig1b, 
                    "Figure 2: Histogram for Raw Lab Data": fig2b,
                    "Figure 3: Normal probability plot of expanded uncertainties based on ranks" : fig3, 
                    "Figure 4: Normal probability plot of expanded uncertainties based on rank percentages" : fig4,
                    "Figure 5: Normal probability plot of result values based with z-score outliers" : fig5,
                    "Figure 6: Bar Chart of Z-scores for proficiency testing" : fig6,
                    "Figure 7: Plot of standard deviations against averages with significance values" : fig7
                   }
    }
    #task = "" ##

    jsonOutput = jsonify( task )

    #debug area
    #######################################################
    # Entire Response JSON

    #xx = open("C:/Python27/delete123.txt", "w")
    #xx.writelines(str(task))
    #xx.close()

    # IMAGE
    """
    decc = base64.standard_b64decode(fig1) 
    mystr = pickle.loads(decc)
    stb = cStringIO.StringIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/interlab_test_image.png', 'png')
    """
    # DELETE THIS ONWARDS
    """
    decc = base64.standard_b64decode(fig1a) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Ilt/fig1aW.png', 'png')
 
    decc = base64.standard_b64decode(fig1b) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Ilt/fig1bW.png', 'png')

    decc = base64.standard_b64decode(fig2a) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Ilt/fig2aW.png', 'png')

    decc = base64.standard_b64decode(fig2b) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Ilt/fig2bW.png', 'png')

    decc = base64.standard_b64decode(fig3) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Ilt/fig3W.png', 'png')

    decc = base64.standard_b64decode(fig4) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Ilt/fig4W.png', 'png')

    decc = base64.standard_b64decode(fig5) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Ilt/fig5W.png', 'png')

    decc = base64.standard_b64decode(fig6) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Ilt/fig6W.png', 'png')

    decc = base64.standard_b64decode(fig7) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Ilt/fig7W.png', 'png')
    """
    #END DELETE THIS
    return jsonOutput, 201 

if __name__ == '__main__': 
    app.run(host="0.0.0.0", port = 5000, debug = True)	

# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/iltW.json http://localhost:5000/pws/interlabtest
# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/ilt1.json http://test.jaqpot.org:8091/pws/interlabtest
# C:\Python27\Flask-0.10.1\python-api 
#C:/Python27/python iltest.py
