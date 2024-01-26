 

import base64
import io
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
import urllib3
from .models import Cake, CakeFinalJson, CakeComment, CakesDataJson
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework  import  viewsets
from  .serializers import CakesSerializer, CakesJsonSerializer, CakesCommentSerializer, CakesDataJsonSerializer
from django.core.mail import EmailMultiAlternatives, send_mail
import json
from django.conf import settings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


#from django_matplotlib.fields import plt
import urllib.parse



#def send_mail_func(request):
#  myemail= send_mail("New Order!", "Hey buddy, you have a new order",
#  "sender-email@gmail.com", ["reciever-email@gmail.com"])
#  template = loader.get_template('email.html')
#  context = {
#    'myemail' : myemail,
#  }
#  return HttpResponse(template.render(context, request))


#def send_mail_func(request):
#    send_mail("New Order!", "Hey buddy, you have a new order",
#          "sender-email@gmail.com", ["reciever-email@gmail.com"])
#    return HttpResponse("Email Sent")


URL = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'

df = pd.read_csv(URL)
df.head()
df_ = df.drop(columns=['species']).copy()


def Gaus_neuron(df, n, step, s):

    neurons_list = list()
    x_axis_list = list()
    t = 0

    for col in df.columns:

        vol = df[col].values
        min_ = np.min(vol)
        max_ = np.max(vol)
        x_axis = np.arange(min_, max_, step)
        x_axis[0] = min_
        x_axis[-1] = max_
        x_axis_list.append(np.round(x_axis, 10))
        neurons = np.zeros((n, len(x_axis)))

        for i in range(n):

            loc = (max_ - min_) * (i /(n-1)) + min_
            neurons[i] = norm.pdf(x_axis, loc, s[t])
            neurons[i] = neurons[i] / np.max(neurons[i])

        neurons_list.append(neurons)
        t += 1

    return neurons_list, x_axis_list

def Lat_Spike(df, d, n):

    for i in range(len(df.columns)):

        k = len(df.iloc[:, i])
        st1 = np.tile(d[1][i], (k, 1))
        st2 = df.iloc[:, i].values.reshape(-1, 1)
        ind = (st1 == st2)
        exc = np.tile(d[0][i], (k, 1)).reshape(k, n, len(d[0][i][0]))[
            np.repeat(ind, n, axis=0).reshape(k, n, len(ind[0]))].reshape(k, n)
        lat_neuron = np.transpose(np.where(exc > 0.1, 1 - exc, np.nan))

        if i == 0:
            lat_neuron_total = lat_neuron
        else:
            lat_neuron_total = np.concatenate((lat_neuron_total, lat_neuron), axis = 0)

    lat_neuron_total[lat_neuron_total == 0] = 0.0001

    return lat_neuron_total

def model_data(ind, ind_type, lat_ne, start, end):
    
    train_stack = np.vstack((lat_ne[ind_type[ind, 0] + start:ind_type[ind, 0] + end],
                            lat_ne[ind_type[ind, 1] + start:ind_type[ind, 1] + end],
                            lat_ne[ind_type[ind, 2] + start:ind_type[ind, 2] + end]))
    train_stack = np.where(train_stack > 0, train_stack, 0)
    
    return train_stack

def LIF_SNN(n, l, data, weight, v_spike):
    
    V_min = 0
    V_spike = v_spike
    r = 5
    tau = 2.5
    dt = 0.01
    t_max = 10
    time_stamps = t_max / dt
    time_relax = 10
    v = np.zeros((n, l, int(time_stamps)))
    t_post = np.zeros((n, l))
    t_post_ = np.zeros((n, int(l / 3)))
    v[:, :, 0] = V_min
    
    for n in range(n):
        for u in range(l):
            
            t = 0
            f0 = (np.round(data[u][np.newaxis].T, 3) * 1000).astype(int)
            f1 = np.tile(np.arange(1000), (40, 1))
            f2 = np.where(((f1 == f0) & (f0 > 0)), 1, 0)
            f2 = f2 * weight[n][np.newaxis].T
            spike_list = np.sum(f2.copy(), axis = 0)

            for step in range(int(time_stamps) - 1):
                if v[n, u, step] > V_spike:
                    t_post[n, u] = step
                    v[n, u, step] = 0
                    t = time_relax / dt
                elif t > 0:
                    v[n, u, step] = 0
                    t = t - 1

                v[n, u, step + 1] = v[n, u, step] + dt / tau * (-v[n, u, step] + r * spike_list[step])
        t_post_[n, :] = t_post[n, n * int(l / 3):n * int(l / 3) + int(l / 3)]
    
    return v, t_post_, t_post

def spike_plot(spike_times, one_per, n, cur_type):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (25, 10))#, dpi = 70)
    
    if one_per:
        k, t, a  = 1, n, 0
        cur = cur_type
    else:
        k, t, a = len(spike_times[0]), 0, 1
        cur = 1
        
    spike_times[spike_times == 0] = np.nan
    di = {0: 'blue', 1: 'red', 2: 'black'}
    di_t = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    p = 0
    
    for ax in [ax1, ax2, ax3]:
        for i in range(k * t, k + t):
            ax.vlines(x = spike_times[p, i] / 100 + i * a * 10, ymin = 0.0, ymax = 1.1, 
                       colors = di[p], ls = '-', lw = 3)
            ax.set_ylabel(f'Neuron {p + 1} \n {di_t[p]}', fontsize = 15)
            
        if one_per:
            ax.axvspan(0, int(k * 10), color = di[cur - 1], alpha = 0.05, label = di_t[cur - 1])
            ax.margins(0)
        else:
            ax.axvspan(0, int(k * 10 / 3), color = di[0], alpha = 0.05, label = di_t[0])
            ax.axvspan(int(k * 10 / 3), int(k * 10 * 2 / 3), color = di[1], alpha = 0.05, label = di_t[1])
            ax.axvspan(int(k * 10 * 2 / 3), int(k * 10 * 3 / 3), color = di[2], alpha = 0.05, label = di_t[2])
            ax.set_xlim(0, k * 10)
            ax.margins(0)
            
        p += 1
        
    
    if one_per:
        plt.suptitle(f' \n\n Moment of spike of postsynaptic neurons for train period {n}', fontsize = 20)
        plt.legend(title = "    Part of a type set:" ,bbox_to_anchor = (1, 1.9), loc = 'upper left',
               fontsize = 15, title_fontsize = 15)
    else:
        plt.suptitle(f' \n\n Moment of spike of postsynaptic neurons on the used part of the dataset', fontsize = 20)
        plt.legend(title = "    Part of a type set:" ,bbox_to_anchor = (1, 2.1), loc = 'upper left',
               fontsize = 15, title_fontsize = 15)
    
    plt.xlabel('Time (ms)', fontsize = 15)
    #plt.show()

def v_plot(v):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (25, 10))#, dpi = 70)
    k = len(v[0,:,:])
    di = {0: 'blue', 1: 'red', 2: 'black'}
    di_t = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    p = 0
    
    for ax in [ax1, ax2, ax3]:
        for i in range(k):
            ax.plot(np.arange(i * 10, (i + 1) * 10, 0.01), v[p, i, :], di[p], linewidth = 1)
            ax.set_ylabel(f' Neuron {p + 1} \n {di_t[p]} \nV (mV)', fontsize = 15)

        ax.axvspan(0, int(k * 10 / 3), color = di[0], alpha = 0.05, label = di_t[0])
        ax.axvspan(int(k * 10 / 3), int(k * 10 * 2 / 3), color = di[1], alpha = 0.05, label = di_t[1])
        ax.axvspan(int(k * 10 * 2 / 3), int(k * 10 * 3 / 3), color = di[2], alpha = 0.05, label = di_t[2])
        ax.margins(0)

        p += 1
    
    plt.legend(title = "    Part of a type set:" ,bbox_to_anchor = (1, 2), loc = 'upper left', fontsize = 15, title_fontsize = 15)
    plt.xlabel('Time (ms)', fontsize = 15)
    plt.suptitle(' \n Activity of postsynaptic neurons on the used part of the dataset \n (Membrane potential)', fontsize = 20)

def accuracy_snn(spike_time, start, end, df, ind_type, ind):
    
    type_dict = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
    target_type_total = np.array(df.replace({'Species': type_dict}).iloc[:, - 1])
    target_type = np.vstack((target_type_total[ind_type[ind, 0] + start:ind_type[ind, 0] + end],
                            target_type_total[ind_type[ind, 1] + start:ind_type[ind, 1] + end],
                            target_type_total[ind_type[ind, 2] + start:ind_type[ind, 2] + end])).flatten()
    
    spike_time_ = np.where(spike_time > 0, np.array(([1], [2], [3])), np.nan)
    final_test = np.full([len(spike_time[0])], np.nan).astype(int)
    for i in range(len(spike_time[0])):
        try:
            final_test[i] = spike_time_[:, i][spike_time[:, i] == np.min(spike_time[:, i][spike_time[:, i] > 0])][0]
        except:
            final_test[i] = 0
    
    ac = np.sum(np.where(final_test == target_type, 1, 0)) / len(target_type)

    return final_test, target_type, print('accur.:', np.round(ac * 100, 2), '%')


def LifModelFunc(self):

  df_.plot.hist(alpha=0.4, figsize=(20, 8))
  plt.legend(title = "Dataset cilumns:" ,bbox_to_anchor = (1.0, 0.6), loc = 'upper left')
  plt.title('Iris dataset', fontsize = 20)
  plt.xlabel('Input value', fontsize = 15)
  plt.show()
 


  return HttpResponse( 'LIF MODEL')


def showDataset(request):
  df_.plot.hist(alpha=0.4, figsize=(20, 8))
  plt.legend(title = "Dataset cilumns:" ,bbox_to_anchor = (1.0, 0.6), loc = 'upper left')
  plt.title('Iris dataset', fontsize = 20)
  plt.xlabel('Input value', fontsize = 15)


  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri =  urllib.parse.quote(string)
  return render(request,'lifmodeldataset.html',{'data':uri})

def receptiveFields(request):
  sigm = [0.1, 0.1, 0.2, 0.1]
  d = Gaus_neuron(df_, 10, 0.001, sigm)

  fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

  fig.set_figheight(8)
  fig.set_figwidth(10)

  k = 0

  for ax in [ax1, ax2, ax3, ax4]:

      ax.set(ylabel = f'{df_.columns[k]} \n\n Excitation of Neuron')

      for i in range(len(d[0][k])):

          ax.plot(d[1][k], d[0][k][i], label = i + 1)

      k+=1

  plt.legend(title = "Presynaptic neuron number \n      in each input column" ,bbox_to_anchor = (1.05, 3.25), loc = 'upper left')
  plt.suptitle(' \n\n  Gaussian receptive fields for Iris dataset', fontsize = 15)
  ax.set_xlabel(' Presynaptic neurons and\n input range of value feature', fontsize = 12, labelpad = 15)

  #plt.show()

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri =  urllib.parse.quote(string)
  fig = plt.gcf()


  x_input = 5
  fig, ax = plt.subplots(1)

  fig.set_figheight(5)
  fig.set_figwidth(15)

  ax.set(ylabel = df_.columns[1])

  for i in range(len(d[0][1])):
      ax.plot(d[1][1], d[0][1][i])

  for n in range(x_input):

      plt.plot(np.tile(df_['sepal_width'][n], (10,1)), 
          d[0][1][np.tile(d[1][1] == df_['sepal_width'][n], (10,1))], 'ro', markersize=4)

      plt.vlines(x = df_['sepal_width'][n], ymin = - 0.1, ymax = 1.1, 
                colors = 'purple', ls = '--', lw = 1, label = df_['sepal_width'][n])

      plt.text(df_['sepal_width'][n] * 0.997, 1.12, n + 1, size = 10)


  plt.legend(title = "First five input:", bbox_to_anchor = (1.0, 0.7), loc = 'upper left')

  plt.suptitle('Gaussian receptive fields for Iris dataset. \n \
                  A detailed description of the idea using the example of the first five value "sepal_width"',
              fontsize = 15)

  ax.set_xlabel('Input value X ∈ [x_min, x_max] of column', fontsize = 12, labelpad = 15)
  ax.set_ylabel('Excitation of a Neuron ∈ [0,1]', fontsize = 12, labelpad = 15)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri3 =  urllib.parse.quote(string)


  return render(request,'fields.html',{'data':uri ,'data3':uri3})

def latancy(request):
  
  sigm = [0.1, 0.1, 0.2, 0.1]
  d = Gaus_neuron(df_, 10, 0.001, sigm)

  x_input = 5

  np.set_printoptions(formatter={'float_kind':'{:f}'.format})
  five_x = np.zeros((5, 10))
  
  for n in range(x_input):
      five_x[n,:] = d[0][1][np.tile(d[1][1] == df_['sepal_width'][n], (10,1))]

  five_x


  five_x = np.where(five_x > 0.1, 1 - five_x, np.nan)
  five_x[five_x == 0] = 0.0001
  five_x


  fig, ax = plt.subplots(5, figsize=(12, 10), dpi = 100)

  for i in range(5):
      ax[i].scatter(x = five_x[i], y = np.arange(1, 10 + 1), s = 10, color = 'black')
      ax[i].hlines(xmin = 0, xmax=1, y=np.arange(1, 11, 1), 
                colors = 'purple', ls = '--', lw = 0.25)
      ax[i].yaxis.set_ticks(np.arange(0, 11, 1))
      ax[i].set_ylabel(f'x{i+1} = {df_.iloc[i,1]}\n (period {i+1}) \n\n № \npre-synaptic neuron')
      ax[i].set_xlim(0, 1)
      ax[i].set_ylim(0, 10 * 1.05)

  ax[i].set_xlabel('Spike Latancy')
  plt.suptitle(' \n\n Input after applying latancy coding \nusing the Gaussian receptive fields method', fontsize = 15)
  #plt.show()


  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri4 =  urllib.parse.quote(string)

  return render(request,'latancy.html',{'data4':uri4})

def latancy2(request):
  sigm = [0.1, 0.1, 0.2, 0.1]
  d = Gaus_neuron(df_, 10, 0.001, sigm)

  fin = Lat_Spike(df_, d, 10)


  fig, ax = plt.subplots(4, figsize=(12, 10), dpi = 100)

  for i in range(4):

      ax[i].scatter(x = fin[i * 10:10 * (1 + i), 0], y = np.arange(1, 10 + 1), s = 10, color = 'r')
      ax[i].hlines(xmin = 0, xmax = 1, y=np.arange(1, 11, 1), 
                colors = 'purple', ls = '--', lw = 0.25)
      ax[i].yaxis.set_ticks(np.arange(0, 11, 1))
      ax[i].set_ylabel(f'col_{i + 1}: {(df_.columns)[i]} \n x1 = {df_.iloc[0, i]} \n (period {1})\n\n № \npre-synaptic neuro')
      ax[i].set_xlim(0, 1)
      ax[i].set_ylim(0, 10 * 1.05)

  ax[i].set_xlabel('Spike Latancy')
  plt.suptitle(' \n\n First input in each column \n after applying latancy coding using the Gaussian receptive fields method', fontsize = 15)
  #plt.show()


  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri5 =  urllib.parse.quote(string)
  return render(request,'latancy2.html',{'data5':uri5})

def presynapneurons(request):
  
  sigm = [0.1, 0.1, 0.2, 0.1]
  d = Gaus_neuron(df_, 10, 0.001, sigm)

  fin = Lat_Spike(df_, d, 10)

  Final_df = pd.DataFrame(fin)
  Final_df

  fig, ax = plt.subplots(1, figsize=(15, 10), dpi = 100)
  h = 3

  for i in range(h):
      ax.scatter(x = (i+Final_df.iloc[:,i].values)*10, y = np.arange(1, 41), s = 8, color = 'black')

      plt.vlines(x = (i)*10, ymin = 0, ymax = 40, 
                colors = 'purple', ls = '--', lw = 1)

  ax.yaxis.set_ticks(np.arange(1, 41, 1))
  ax.xaxis.set_ticks(np.arange(0, (h+1)*10, 10))
  ax.set_xlabel('time (ms)')
  ax.set_ylabel('№ presynaptic neuron')
  plt.suptitle(' \n\n\n Spikes of presynaptic neurons for first 30 ms', fontsize = 15)
  plt.gca().invert_yaxis()
  #plt.show()


  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri6 =  urllib.parse.quote(string)
  return render(request,'presynapneurons.html',{'data6':uri6})


def postsynaptic(request):
  sigm = [0.1, 0.1, 0.2, 0.1]
  d = Gaus_neuron(df_, 10, 0.001, sigm)

  fin = Lat_Spike(df_, d, 10)

  Final_df = pd.DataFrame(fin)
  Final_df
  lat_ne = np.transpose(Final_df.values)
  ind_type = np.array(([0, 50, 100], [50, 100, 0], [100, 0, 50]))
  list_weight = np.zeros((3,40))

  for ind in range(3):
      
      train_stack = model_data(ind, ind_type, lat_ne, 0, 20)
      tr_ar = np.where(np.transpose(train_stack) > 0, 2 * (1 - np.transpose(train_stack)), 0)
      tr_ar[:, 20:] = tr_ar[:, 20:] * (-1)
      tr_ar = pd.DataFrame(tr_ar)
      tr_ar[20] = tr_ar.iloc[:,:20].sum(axis = 1) + 0.1
      tst_ar = np.float64(np.transpose(np.array(tr_ar.iloc[:,20:])))
      
      for i in range(1, len(tst_ar)):
          
          tst_ar[0][((np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))] += - np.float64(
              np.sum(tst_ar[i][np.round(tst_ar[0], 4) > 0.1]) / len(tst_ar[0][((
                  np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))]))
          tst_ar[0][np.round(tst_ar[0], 4) > 0.1] += tst_ar[i][np.round(tst_ar[0], 4) > 0.1]
          tst_ar[0][tst_ar[0] < 0.1] = 0.1
          
      list_weight[ind, :] = tst_ar[0]

  list_weight


  train_stack = model_data(0, ind_type, lat_ne, 0, 20)
  res = LIF_SNN(3, 60, train_stack, list_weight, 100)
  v = res[0]

  v_plot(v)
  
  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri7 =  urllib.parse.quote(string)

  res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
  spike_time = res[2]
  spike_plot(spike_time, False, False, False)
  
  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri8 =  urllib.parse.quote(string)

  accuracy_snn(spike_time, 0, 20, df, ind_type, 0)[2]


  spike_plot(spike_time, True, 46, 3)
  
  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri9 =  urllib.parse.quote(string)  

  spike_plot(spike_time, True, 24, 2)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri10 =  urllib.parse.quote(string)  



  train_stack = model_data(0, ind_type, lat_ne, 20, 40)
  res = LIF_SNN(3, 60, train_stack, list_weight, 100)
  v = res[0]

  v_plot(v)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri11 =  urllib.parse.quote(string)  


  res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
  spike_time = res[2]
  spike_plot(spike_time, False, False, False)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri12 =  urllib.parse.quote(string)  



  return render(request,'postsynaptic.html',{'data7':uri7, 'data7':uri8, 'data9':uri9,'data10':uri10, 'data11':uri11, 'data12':uri12})

def synapticspikes(request):
  
  sigm = [0.1, 0.1, 0.2, 0.1]
  d = Gaus_neuron(df_, 10, 0.001, sigm)

  fin = Lat_Spike(df_, d, 10)

  Final_df = pd.DataFrame(fin)
  Final_df

  lat_ne = np.transpose(Final_df.values)
  ind_type = np.array(([0, 50, 100], [50, 100, 0], [100, 0, 50]))

  list_weight = np.zeros((3,40))

  for ind in range(3):
      
      train_stack = model_data(ind, ind_type, lat_ne, 0, 20)
      tr_ar = np.where(np.transpose(train_stack) > 0, 2 * (1 - np.transpose(train_stack)), 0)
      tr_ar[:, 20:] = tr_ar[:, 20:] * (-1)
      tr_ar = pd.DataFrame(tr_ar)
      tr_ar[20] = tr_ar.iloc[:,:20].sum(axis = 1) + 0.1
      tst_ar = np.float64(np.transpose(np.array(tr_ar.iloc[:,20:])))
      
      for i in range(1, len(tst_ar)):
          
          tst_ar[0][((np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))] += - np.float64(
              np.sum(tst_ar[i][np.round(tst_ar[0], 4) > 0.1]) / len(tst_ar[0][((
                  np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))]))
          tst_ar[0][np.round(tst_ar[0], 4) > 0.1] += tst_ar[i][np.round(tst_ar[0], 4) > 0.1]
          tst_ar[0][tst_ar[0] < 0.1] = 0.1
          
      list_weight[ind, :] = tst_ar[0]

  list_weight





  res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
  spike_time = res[2]

  accuracy_snn(spike_time, 20, 40, df, ind_type, 0)[2]

  train_stack = model_data(0, ind_type, lat_ne, 20, 40)


  res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
  t_post = res[1]
  A_p = 0.8
  A_m = A_p * 1.1

  for n in range(3):
      for u in range(20):
          
          t1 = np.round(train_stack[u + 10 * n] * 1000)
          t2 = t1.copy()
          
          t2[((t1 <= t_post[n, u]) & (t1 > 0))] = A_p * np.exp((t1[((t1 <= t_post[n, u]) & (t1 > 0))] - t_post[n, u]) / 1000)
          t2[((t1 > t_post[n, u]) & (t1 > 0))] = - A_m * np.exp((t_post[n, u] - t1[((t1 > t_post[n, u]) & (t1 > 0))]) / 1000)
          
          list_weight[n, :] += t2
          
  list_weight[list_weight < 0] = 0
  list_weight

  res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
  spike_time = res[2]
  spike_plot(spike_time, False, False, False)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri13 =  urllib.parse.quote(string)  


  accuracy_snn(spike_time, 20, 40, df, ind_type, 0)[2]


  train_stack = model_data(0, ind_type, lat_ne, 0, 40)
  res = LIF_SNN(3, 120, train_stack, list_weight, 100)
  v = res[0]

  v_plot(v)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri14 =  urllib.parse.quote(string)  


  res = LIF_SNN(3, 120, train_stack, list_weight, 0.25)
  spike_time = res[2]
  spike_plot(spike_time, False, False, False)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri15 =  urllib.parse.quote(string)  

  accuracy_snn(spike_time, 0, 40, df, ind_type, 0)[2]


  train_stack = model_data(0, ind_type, lat_ne, 40, 50)
  res = LIF_SNN(3, 30, train_stack, list_weight, 100)
  v = res[0]
  res = LIF_SNN(3, 30, train_stack, list_weight, 0.25)
  spike_time = res[2]

  v_plot(v)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri16 =  urllib.parse.quote(string)  



  spike_plot(spike_time, False, False, False)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri17 =  urllib.parse.quote(string)  


  accuracy_snn(spike_time, 40, 50, df, ind_type, 0)[2]


  spike_plot(spike_time, True, 27, 3)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri18 =  urllib.parse.quote(string)  




  return render(request,'synapticspikes.html',{'data13':uri13, 'data14':uri14, 'data15':uri15, 'data16':uri16, 'data17':uri17, 'data18':uri18})


def callNeuralNets(request):
  ''' mymember = "this is the context"
  template = loader.get_template('lifmodel.html')
  context = {
    'cakescomymembermment' : mymember,
    }
  return HttpResponse(template.render(context, request))
  '''
  df_.plot.hist(alpha=0.4, figsize=(20, 8))
  plt.legend(title = "Dataset cilumns:" ,bbox_to_anchor = (1.0, 0.6), loc = 'upper left')
  plt.title('Iris dataset', fontsize = 20)
  plt.xlabel('Input value', fontsize = 15)


  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri =  urllib.parse.quote(string)

  sigm = [0.1, 0.1, 0.2, 0.1]
  d = Gaus_neuron(df_, 10, 0.001, sigm)

  fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

  fig.set_figheight(8)
  fig.set_figwidth(10)

  k = 0

  for ax in [ax1, ax2, ax3, ax4]:

      ax.set(ylabel = f'{df_.columns[k]} \n\n Excitation of Neuron')

      for i in range(len(d[0][k])):

          ax.plot(d[1][k], d[0][k][i], label = i + 1)

      k+=1

  plt.legend(title = "Presynaptic neuron number \n      in each input column" ,bbox_to_anchor = (1.05, 3.25), loc = 'upper left')
  plt.suptitle(' \n\n  Gaussian receptive fields for Iris dataset', fontsize = 15)
  ax.set_xlabel(' Presynaptic neurons and\n input range of value feature', fontsize = 12, labelpad = 15)


  #plt.show()

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri2 =  urllib.parse.quote(string)




  x_input = 5
  fig, ax = plt.subplots(1)

  fig.set_figheight(5)
  fig.set_figwidth(15)

  ax.set(ylabel = df_.columns[1])

  for i in range(len(d[0][1])):
      ax.plot(d[1][1], d[0][1][i])

  for n in range(x_input):

      plt.plot(np.tile(df_['sepal_width'][n], (10,1)), 
          d[0][1][np.tile(d[1][1] == df_['sepal_width'][n], (10,1))], 'ro', markersize=4)

      plt.vlines(x = df_['sepal_width'][n], ymin = - 0.1, ymax = 1.1, 
                colors = 'purple', ls = '--', lw = 1, label = df_['sepal_width'][n])

      plt.text(df_['sepal_width'][n] * 0.997, 1.12, n + 1, size = 10)


  plt.legend(title = "First five input:", bbox_to_anchor = (1.0, 0.7), loc = 'upper left')

  plt.suptitle('Gaussian receptive fields for Iris dataset. \n \
                  A detailed description of the idea using the example of the first five value "sepal_width"',
              fontsize = 15)

  ax.set_xlabel('Input value X ∈ [x_min, x_max] of column', fontsize = 12, labelpad = 15)
  ax.set_ylabel('Excitation of a Neuron ∈ [0,1]', fontsize = 12, labelpad = 15)

  #plt.show()


  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri3 =  urllib.parse.quote(string)




  np.set_printoptions(formatter={'float_kind':'{:f}'.format})
  five_x = np.zeros((5, 10))

  for n in range(x_input):
      five_x[n,:] = d[0][1][np.tile(d[1][1] == df_['sepal_width'][n], (10,1))]

  five_x


  five_x = np.where(five_x > 0.1, 1 - five_x, np.nan)
  five_x[five_x == 0] = 0.0001
  five_x


  fig, ax = plt.subplots(5, figsize=(12, 10), dpi = 100)

  for i in range(5):
      ax[i].scatter(x = five_x[i], y = np.arange(1, 10 + 1), s = 10, color = 'black')
      ax[i].hlines(xmin = 0, xmax=1, y=np.arange(1, 11, 1), 
                colors = 'purple', ls = '--', lw = 0.25)
      ax[i].yaxis.set_ticks(np.arange(0, 11, 1))
      ax[i].set_ylabel(f'x{i+1} = {df_.iloc[i,1]}\n (period {i+1}) \n\n № \npre-synaptic neuron')
      ax[i].set_xlim(0, 1)
      ax[i].set_ylim(0, 10 * 1.05)

  ax[i].set_xlabel('Spike Latancy')
  plt.suptitle(' \n\n Input after applying latancy coding \nusing the Gaussian receptive fields method', fontsize = 15)
  #plt.show()


  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri4 =  urllib.parse.quote(string)



  fin = Lat_Spike(df_, d, 10)


  fig, ax = plt.subplots(4, figsize=(12, 10), dpi = 100)

  for i in range(4):

      ax[i].scatter(x = fin[i * 10:10 * (1 + i), 0], y = np.arange(1, 10 + 1), s = 10, color = 'r')
      ax[i].hlines(xmin = 0, xmax = 1, y=np.arange(1, 11, 1), 
                colors = 'purple', ls = '--', lw = 0.25)
      ax[i].yaxis.set_ticks(np.arange(0, 11, 1))
      ax[i].set_ylabel(f'col_{i + 1}: {(df_.columns)[i]} \n x1 = {df_.iloc[0, i]} \n (period {1})\n\n № \npre-synaptic neuro')
      ax[i].set_xlim(0, 1)
      ax[i].set_ylim(0, 10 * 1.05)

  ax[i].set_xlabel('Spike Latancy')
  plt.suptitle(' \n\n First input in each column \n after applying latancy coding using the Gaussian receptive fields method', fontsize = 15)
  #plt.show()


  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri5 =  urllib.parse.quote(string)


  Final_df = pd.DataFrame(fin)
  Final_df


  fig, ax = plt.subplots(1, figsize=(15, 10), dpi = 100)
  h = 3

  for i in range(h):
      ax.scatter(x = (i+Final_df.iloc[:,i].values)*10, y = np.arange(1, 41), s = 8, color = 'black')

      plt.vlines(x = (i)*10, ymin = 0, ymax = 40, 
                colors = 'purple', ls = '--', lw = 1)

  ax.yaxis.set_ticks(np.arange(1, 41, 1))
  ax.xaxis.set_ticks(np.arange(0, (h+1)*10, 10))
  ax.set_xlabel('time (ms)')
  ax.set_ylabel('№ presynaptic neuron')
  plt.suptitle(' \n\n\n Spikes of presynaptic neurons for first 30 ms', fontsize = 15)
  plt.gca().invert_yaxis()
  #plt.show()



  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri6 =  urllib.parse.quote(string)


  lat_ne = np.transpose(Final_df.values)
  ind_type = np.array(([0, 50, 100], [50, 100, 0], [100, 0, 50]))
  list_weight = np.zeros((3,40))

  for ind in range(3):
      
      train_stack = model_data(ind, ind_type, lat_ne, 0, 20)
      tr_ar = np.where(np.transpose(train_stack) > 0, 2 * (1 - np.transpose(train_stack)), 0)
      tr_ar[:, 20:] = tr_ar[:, 20:] * (-1)
      tr_ar = pd.DataFrame(tr_ar)
      tr_ar[20] = tr_ar.iloc[:,:20].sum(axis = 1) + 0.1
      tst_ar = np.float64(np.transpose(np.array(tr_ar.iloc[:,20:])))
      
      for i in range(1, len(tst_ar)):
          
          tst_ar[0][((np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))] += - np.float64(
              np.sum(tst_ar[i][np.round(tst_ar[0], 4) > 0.1]) / len(tst_ar[0][((
                  np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))]))
          tst_ar[0][np.round(tst_ar[0], 4) > 0.1] += tst_ar[i][np.round(tst_ar[0], 4) > 0.1]
          tst_ar[0][tst_ar[0] < 0.1] = 0.1
          
      list_weight[ind, :] = tst_ar[0]

  list_weight


  train_stack = model_data(0, ind_type, lat_ne, 0, 20)
  res = LIF_SNN(3, 60, train_stack, list_weight, 100)
  v = res[0]

  v_plot(v)
  
  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri7 =  urllib.parse.quote(string)

  res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
  spike_time = res[2]
  spike_plot(spike_time, False, False, False)
  
  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri8 =  urllib.parse.quote(string)

  accuracy_snn(spike_time, 0, 20, df, ind_type, 0)[2]


  spike_plot(spike_time, True, 46, 3)
  
  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri9 =  urllib.parse.quote(string)  

  spike_plot(spike_time, True, 24, 2)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri10 =  urllib.parse.quote(string)  



  train_stack = model_data(0, ind_type, lat_ne, 20, 40)
  res = LIF_SNN(3, 60, train_stack, list_weight, 100)
  v = res[0]

  v_plot(v)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri11 =  urllib.parse.quote(string)  


  res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
  spike_time = res[2]
  spike_plot(spike_time, False, False, False)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri12 =  urllib.parse.quote(string)  

  accuracy_snn(spike_time, 20, 40, df, ind_type, 0)[2]

  res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
  t_post = res[1]
  A_p = 0.8
  A_m = A_p * 1.1

  for n in range(3):
      for u in range(20):
          
          t1 = np.round(train_stack[u + 10 * n] * 1000)
          t2 = t1.copy()
          
          t2[((t1 <= t_post[n, u]) & (t1 > 0))] = A_p * np.exp((t1[((t1 <= t_post[n, u]) & (t1 > 0))] - t_post[n, u]) / 1000)
          t2[((t1 > t_post[n, u]) & (t1 > 0))] = - A_m * np.exp((t_post[n, u] - t1[((t1 > t_post[n, u]) & (t1 > 0))]) / 1000)
          
          list_weight[n, :] += t2
          
  list_weight[list_weight < 0] = 0
  list_weight

  res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
  spike_time = res[2]
  spike_plot(spike_time, False, False, False)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri13 =  urllib.parse.quote(string)  

  accuracy_snn(spike_time, 20, 40, df, ind_type, 0)[2]


  train_stack = model_data(0, ind_type, lat_ne, 0, 40)
  res = LIF_SNN(3, 120, train_stack, list_weight, 100)
  v = res[0]

  v_plot(v)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri14 =  urllib.parse.quote(string)  


  res = LIF_SNN(3, 120, train_stack, list_weight, 0.25)
  spike_time = res[2]
  spike_plot(spike_time, False, False, False)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri15 =  urllib.parse.quote(string)  

  accuracy_snn(spike_time, 0, 40, df, ind_type, 0)[2]


  train_stack = model_data(0, ind_type, lat_ne, 40, 50)
  res = LIF_SNN(3, 30, train_stack, list_weight, 100)
  v = res[0]
  res = LIF_SNN(3, 30, train_stack, list_weight, 0.25)
  spike_time = res[2]

  v_plot(v)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri16 =  urllib.parse.quote(string)  



  spike_plot(spike_time, False, False, False)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri17 =  urllib.parse.quote(string)  


  accuracy_snn(spike_time, 40, 50, df, ind_type, 0)[2]


  spike_plot(spike_time, True, 27, 3)

  fig = plt.gcf()
  #convert graph into dtring buffer and then we convert 64 bit code into image
  buf = io.BytesIO()
  fig.savefig(buf,format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri18 =  urllib.parse.quote(string)  



  return render(request,'lifmodel.html',{'data':uri, 'data2': uri2, 'data3': uri3, 'data4': uri4, 'data5': uri5, 'data6': uri6, 'data7': uri7, 'data8': uri8, 'data9': uri9
                                         , 'data10': uri10, 'data11': uri11, 'data12': uri12, 'data13': uri13, 'data14': uri14, 'data15': uri15, 'data16': uri16, 'data17': uri17
                                         , 'data18': uri18})




'''
 {
    "name": "Mac\u00edas P", 
    "email": "maciasp05@gmail.com",
   "ingredients": ["Vailla fresas"], 
   "type": "Clasico",
    "size": "mediano",
    "event": "Navidad", 
    "shape": "Doble Redondo",
    "address": "Lomas lejos",
    "designdraw": "Cl\u00e1sico con flores",
    "textcake": "Feliz Navidad",
    "notes": "Sin notas adiconales",
    "deliverydate": "January 31, 2024-04:00 PM",
    "cellphone": "6643753412",
    "uniqueId": "6yal984n4"}

'''


def send_welcome_email(jsondata, email):

    workorder ="" 
    name = ""
    emailval = ""
   #ingredients =[]
    type = ""
    size = ""
    event = ""
    shape = ""
    address = ""
    designdraw = ""
    textcake = ""
    notes = ""
    deliverydate = ""
    ingreStr = ""
    cellphone = ""
    for key, value in jsondata.items():
      print(key, value)
      if (key == "uniqueId"):
          workorder = value
      if (key == "name"):
          name = value
      if (key == "email"):
          emailval = value
      if (key == "ingredients"):
          for x in value:
            ingreStr= ingreStr + x + " "
            print(ingreStr)
      if (key == "type"):
          type = value
      if (key == "size"):
          size = value
      if (key == "event"):
          event = value  
      if (key == "shape"):
          shape = value  
      if (key == "address"):
          address = value  
      if (key == "designdraw"):
          designdraw = value  
      if (key == "textcake"):
          textcake = value  
      if (key == "notes"):
          notes = value  
      if (key == "deliverydate"):
          deliverydate = value  
      if (key == "cellphone"):
          cellphone = value  

    subject = 'Your Order is here- Tu Orden esta aqui - Welcome to My Cakes Site'
   # message = "Your current data is : " +json.dumps(jsondata)  + "\n\n" +  "Your workOrder is: "+workorder  +"\n\n For details call or whatsapp to Riky 6641268391"
    message = "Hi "+name+ "\n\nyour info is here/tu info esta aqui!\n INGREDIENTS:\n" + ingreStr +"\n"+ type + "\n" + size + "\n"+ event +"\n"+ shape + "\nDESIGN/DISENO: "+ designdraw + "\nTEXT OVER THE CAKE/TEXTO EN EL PASTEL: "+ textcake+ "\nADDRESS/DIRECCION: "+ address+ "\nNOTES/NOTAS: "+ notes + "\nDELIVERY DATE/FECHA DE ENTREGA: "+deliverydate + "\n\nYOUR EMAIL: "+emailval+"\n\nYOUR CELLPHONE/CELULAR: "+cellphone+"\n\n\nYour workOrder is: "+workorder  +"\n\n\nFor details call or whatsapp to/para mas detalles llamar a  Riky 6641268391"
    from_email = 'ramo2884@gmail.com'
    recipient_list = [json.dumps(email),"ramo2884@gmail.com"]
    send_mail(subject, message, from_email, recipient_list)


    #return HttpResponse("Email Sent", request)
  #  return HttpResponse("Email Sent", request)



# create a viewset
class EmailViewSet(viewsets.ModelViewSet):
   varia ='email was sent'

def cakes(request):
  mycakes= Cake.objects.all().values()
  template = loader.get_template('all_cakes.html')
  context = {
    'mycakes' : mycakes,
  }
  return HttpResponse(template.render(context, request))

# create a viewset
class CakesViewSet(viewsets.ModelViewSet):
    # define queryset
    queryset = Cake.objects.all()
 
    # specify serializer to be used
    serializer_class = CakesSerializer

def cakesJson(request):
  mycakesjson= CakeFinalJson.objects.all().values()
  template = loader.get_template('all_cakesjson.html')
  context = {
    'mycakesjson' : mycakesjson,
  }
  return HttpResponse(template.render(context, request))

class CakesJsonViewSet(viewsets.ModelViewSet):
    # define queryset
    queryset = CakeFinalJson.objects.all()
 
    # specify serializer to be used
    serializer_class = CakesJsonSerializer

def cakesdataJson(request):
  mycakesdatajson= CakesDataJson.objects.all().values()
  template = loader.get_template('all_cakesdatajson.html')
  context = {
    'mycakesdatajson' : mycakesdatajson,
  }
  return HttpResponse(template.render(context, request))

class CakesDataJsonViewSet(viewsets.ModelViewSet):
    # define queryset
    queryset = CakesDataJson.objects.all()
 
    # specify serializer to be used
    serializer_class = CakesDataJsonSerializer

@api_view(['POST'])
def postJsonCake(request):
    serializer = CakesJsonSerializer(data=request.data)
 
    if serializer.is_valid():
        serializer.save()
        json  = serializer.data['jsondata']
        email  = serializer.data['email']
        send_welcome_email(json, email)
        print(json)
        print(email)
    return Response(serializer.data)

def cakesComment(request):
  cakescomment = CakeComment.objects.all().values()
  template = loader.get_template('all_cakescomments.html')
  context = {
    'cakescomment' : cakescomment,
    }
  return HttpResponse(template.render(context, request))

class CakesCommentViewSet(viewsets.ModelViewSet):
    # define queryset
    queryset = CakeComment.objects.all()
 
    # specify serializer to be used
    serializer_class = CakesCommentSerializer

@api_view(['POST'])
def postCommentCake(request):
    serializer = CakesCommentSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
    return Response(serializer.data)




