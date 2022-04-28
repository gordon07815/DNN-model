#https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html#lifelines.fitters.coxph_fitter.CoxPHFitter
#https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import auc
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import re
#from patsy import dmatrices
#import statsmodels.api as sm
#from statsmodels.stats.outliers_influence import variance_inflation_factor

ahf_prog = pd.read_excel('AHF_outcome.xlsx')
ahf_prog['V2日期'][393]='2018/1/1'
eve = pd.DataFrame([[0]*431]*6)

for i in range(431):
    if(type(ahf_prog['Event 1'].astype('string')[i]) != pd._libs.missing.NAType):
        eve.iloc[0][i] = 1
    if(type(ahf_prog['Event 2'].astype('string')[i]) != pd._libs.missing.NAType):
        eve.iloc[1][i] = 1
    if(type(ahf_prog['Event 3'].astype('string')[i]) != pd._libs.missing.NAType):
        eve.iloc[2][i] = 1
    if(type(ahf_prog['Event 4'].astype('string')[i]) != pd._libs.missing.NAType):
        eve.iloc[3][i] = 1
    if(str(ahf_prog['Mortality date'][i])!='nan'):
        eve.iloc[5][i] = 1
        a = re.match(r'(\d+)\D*(\d*)\D*(\d*)', str(ahf_prog['Mortality date'][i]))
        b = re.match(r'(\d+)\D*(\d*)\D*(\d*)', str(ahf_prog['V2日期'][i]))
        if(a.group(2)==''):
            eve.iloc[4][i] = (date(int(a.group(1)), 12, 31) - date(int(b.group(1)), int(b.group(2)), int(b.group(3)))).days
        elif(a.group(3)==''):
            eve.iloc[4][i] = (date(int(a.group(1)), int(a.group(2)), 31) - date(int(b.group(1)), int(b.group(2)), int(b.group(3)))).days
        else:
            eve.iloc[4][i] = (date(int(a.group(1)), int(a.group(2)), int(a.group(3))) - date(int(b.group(1)), int(b.group(2)), int(b.group(3)))).days
    else:
        b = re.match(r'(\d+)\D*(\d*)\D*(\d*)', str(ahf_prog['V2日期'][i]))
        eve.iloc[4][i] = (date(2021, 12, 10) - date(int(b.group(1)), int(b.group(2)), int(b.group(3)))).days

    

ahf_data = pd.read_excel("V2 lab_modified.xlsx")
age_ahf = []
for i in range(431):
    if(str(ahf_data.檢查日[i]).find("NaT") >= 0):
        if(str(ahf_data.出生日[i]).find("NaT") >= 0):
            age_ahf = np.append(age_ahf, "60")
        else:
            age_ahf = np.append(age_ahf, int(2007) - int(str(ahf_data.出生日[i])[:4]))
    elif(str(ahf_data.出生日[i]).find("NaT") >= 0):
        age_ahf = np.append(age_ahf, "60")
    else:
        age_ahf = np.append(age_ahf, int(str(ahf_data.檢查日[i])[:4]) - int(str(ahf_data.出生日[i])[:4]))
age_ahf = age_ahf.reshape(431,1)
sex_ahf = ahf_data.sex.str.replace('女','2')
sex_ahf = sex_ahf.str.replace('男','1')
sex_ahf = sex_ahf.str.replace('M','1')
#sex_ahf = np.array(sex_ahf.str.replace(r'\D+','0',regex=True)).reshape(431,1)
for i in range(len(sex_ahf)):
    if(isinstance(sex_ahf.iloc[i], str)==False and math.isnan(sex_ahf.iloc[i])):
        print(i)
        sex_ahf.iloc[i] = '0'
age_ahf = np.concatenate((age_ahf, np.asarray(sex_ahf[:431]).reshape(431,1)), axis=1)
#age_ahf = np.concatenate((age_ahf, np.array([1]*433).reshape(433,1)), axis=1)
V2 = ahf_data.iloc[:,5:]
V2 = V2.replace({True:1, False:0, '%':''})
V2 = V2[:431].T.append(eve.iloc[4:]).T
V2 = V2.rename({4:'duration', 5:'event'}, axis='columns')

d=pd.DataFrame(range(431))
for i in range(84):
    num = 0
    tot = 0
    pos = []
    for j in range(431):            
        if V2.iloc[j,i]=='UNK' or V2.iloc[j,i]=='preserved' or V2.iloc[j,i]=='NO':
            V2.iloc[j,i]=float('NaN')
        if(isinstance(V2.iloc[j,i],str) or np.isnan(V2.iloc[j,i])==False):
            if re.match(r'\S*[\u4e00-\u9fff]+', str(V2.iloc[j,i])):
                V2.iloc[j,i] = re.match(r'(\S*)[\u4e00-\u9fff]+', V2.iloc[j,i]).group(1)
            if re.match(r'\S*[\u4e00-\u9fff]+', str(V2.iloc[j,i])):
                V2.iloc[j,i] = re.match(r'(\S*)[\u4e00-\u9fff]+', V2.iloc[j,i]).group(1)
            if re.match(r'\d+.*\d*[-~]+\d+.*\d*', str(V2.iloc[j,i])):
                a = re.match(r'(\d+.*\d*)[-~]+(\d+.*\d*)', V2.iloc[j,i])
                V2.iloc[j,i] = np.average([float(a.group(1)), float(a.group(2))])
            if re.match(r'<\d*\.*\d*', str(V2.iloc[j,i])):
                V2.iloc[j,i] = re.match(r'<(\d*\.*\d*)', V2.iloc[j,i]).group(1)
            if re.match(r'(\S*)\+', str(V2.iloc[j,i])):
                V2.iloc[j,i] = re.match(r'(\S*)\+', str(V2.iloc[j,i])).group(1)
            num = num + 1
            V2.iloc[j,i] = re.match(r'(\d*\.*\d*)', str(V2.iloc[j,i])).group(1)
            tot = tot + float(V2.iloc[j,i])
        else:
            pos.append(j)
    values=V2.iloc[~d.index.isin(pos),i].astype('float64')
    #print(V2.columns[i]+': %.2f(%.3f)'%(np.average(values), np.std(values)))
    V2.iloc[pos,i]=tot/num


'''
AHF_data = pd.read_csv('AHF_noted.csv', header=None)
V=np.array([0]*8)
#AHF_data_modi = pd.DataFrame()
for i in range(len(AHF_data)):
    if(re.search(r'^AHF\d*V1',AHF_data.iloc[i,0])):
        V[0]=V[0]+1
    elif(re.search(r'^AHF\d*V2',AHF_data.iloc[i,0])):
        #AHF_data_modi = AHF_data_modi.append(AHF_data.iloc[i,1:1000])
        V[1]=V[1]+1
    elif(re.search(r'^AHF\d*V3',AHF_data.iloc[i,0])):
        V[2]=V[2]+1
    elif(re.search(r'^AHF\d*V4',AHF_data.iloc[i,0])):
        V[3]=V[3]+1
    elif(re.search(r'^AHF\d*V5',AHF_data.iloc[i,0])):
        V[4]=V[4]+1
    elif(re.search(r'^AHF\d*V6',AHF_data.iloc[i,0])):
        V[5]=V[5]+1
    elif(re.search(r'^AHF\d*V7',AHF_data.iloc[i,0])):
        V[6]=V[6]+1
    elif(re.search(r'^AH[Ff]\d*V0*8',AHF_data.iloc[i,0])):
        V[7]=V[7]+1
    else:
        print(AHF_data.iloc[i,0])
#AHF_found = AHF_data.iloc[AHF_data_modi.index,0]



AHF_data = pd.concat([AHF_data[range(1000)], AHF_data[1002]],axis=1)
AHF_data= AHF_data.rename(columns={1002:1000})
array = np.empty([0,103])
for i in range(len(AHF_data)):
    if(re.search(r'AHF\d*V2',AHF_data[0][i])):
        a = re.search(r'AHF(\d*)',AHF_data[0][i])
        n = a.group(1)
        if(ahf_data['編號'].str.contains(a.group(0)).any()):
            b = np.append(AHF_data.iloc[i,range(1,990,10)], age_ahf[pd.Index(ahf_data['編號']).get_loc(a.group(0))])[:101].reshape(1,101)
            b = np.append(b, np.array(eve)[4:,int(n)-1])
            array = np.append(array,[b],axis=0)
array = pd.DataFrame(array)
array = pd.concat([array[range(99)], array[[101,102]]],axis=1)

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from patsy import dmatrices, dmatrix
V2_modified =V2.iloc[:,[0,13,18,19,24,28,29,31,39,56,59,63,70,72,74,79,81]]
V2_modified =V2.iloc[:,50:60]
V2_modified=pd.concat([V2_modified, V2.iloc[:,30:60]], axis=1)
fea=V2.columns[[0,13,18,19,24,28,29,31,39,56,59,63,70,72,74,79,81]]
fea=np.append(fea,V2.columns[30:54])
fea=np.append(fea,V2.columns[63:64])
#fea=V2.columns[60:63]
features=array.columns[:-1]
features = "+".join(features[:-1])
y, X = dmatrices('duration ~ ' + features, array, return_type = 'dataframe')
VIF = pd.DataFrame()
VIF['VIF factor'] = [vif(X.values, i) for i in range(X.shape[1])]
VIF["features"] = X.columns
VIF
VIF['features'].iloc[np.where(VIF['VIF factor']>2)]

np.where(V2.columns=='Nt_pro_BNP')
-log2(p)>1.0: [13,18,19,24,28,29,31,59,72,79,81]
-log2(p)>2.0: [13,18,24,28,31,59,72,81]
-log2(p)>4.3: [13, 72, 18, 68, 84,85], [17], [18], [20], [36], [50], [54], [55], [56], 
[57], [58], [60], [61], [62], [64], [68], [69], [73], [76]
'''
AHF_score = pd.read_csv('AHF_score.csv', header=None)
V2_modified = V2.iloc[:,:84]
features=np.append(V2_modified.columns, ['AHF', 'Age', 'Sex', 'duration', 'event'])
array = np.empty([0,89])
ind = []
for i in range(len(AHF_score)):
    a = re.search(r'AHF(\d*)',AHF_score.iloc[i,0])
    n = a.group(1)
    if(ahf_data['編號'].str.contains(a.group(0)).any() and int(n)<=431):
        b = np.append(AHF_score.iloc[i,1], age_ahf[pd.Index(ahf_data['編號']).get_loc(a.group(0))])[:3].reshape(1,3)
        b = np.append(b, np.array(eve)[4:,int(n)-1])
        b = np.append(V2_modified.iloc[int(n)-1], b)
        array = np.append(array,[b],axis=0)
        ind.append(a.group(0))
    else:
        print(a.group(0))
array = pd.DataFrame(array, index=ind, columns=features)

'''a = [0]*6
a[0]=sum(array['age'].astype('float64')<50)
for i in range(1,6):
    a[i]=sum(array['age'].astype('float64')<50+i*10)-sum(array['age'].astype('float64')<40+i*10)
'''

#[1,2,4,10,14,15,16,18,19,20,25,26,27,28,30,31]
#2022/01/05
#array_one = pd.concat([array['age'],array.iloc[:,5:]], axis=1)
cph = CoxPHFitter(penalizer=0.0)
d=[13, 72, 17, 18, 20,36,50,54,55,56,57,58,60,61,62,64,68,69,73, 76, 84,85]
#for i in range(len(d)):
    #if i!=41 and i!=42 and i!=43:
cph.fit(array.iloc[:,[13, 72, 17, 18, 20,36,50,54,55,56,57,58,60,61,62,64,68,69,73, 76, 84,85,87,88]], duration_col='duration', event_col='event')#41,43 high collinearity, 75>>0.85
cph.plot()
plt.show()
cph.print_summary()
'''
    a=cph.summary
    if i ==0:
        w=pd.DataFrame([], columns=a.columns)
    w=pd.concat([w,a])
w.to_csv('Univariate.csv')
'''
#https://medium.com/the-researchers-guide/survival-analysis-in-python-km-estimate-cox-ph-and-aft-model-5533843c5d5d
fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize=(20,20))
cph.plot_partial_effects_on_outcome(covariates='AHF', values=['%.1f'%i for i in np.arange(0,1.1,0.1)], cmap='coolwarm', ax=axs[0][0])
axs[0][0].set_title('AHF score')
cph.plot_partial_effects_on_outcome(covariates='PAD', values=[0, 1], cmap='coolwarm', ax=axs[0][1])
axs[0][1].set_title('PAD')
cph.plot_partial_effects_on_outcome(covariates='PCI', values=[0, 1], cmap='coolwarm', ax=axs[1][0])
axs[1][0].set_title('PCI')
cph.plot_partial_effects_on_outcome(covariates='Age', values=['%d'%i for i in np.arange(40,110,10)], cmap='coolwarm', ax=axs[1][1])
axs[1][1].set_title('Age')
cph.plot_partial_effects_on_outcome(covariates='BUN', values=['%d'%i for i in np.arange(0,160,20)], cmap='coolwarm', ax=axs[2][0])
axs[2][0].set_title('BUN')
cph.plot_partial_effects_on_outcome(covariates='Nt_pro_BNP', values=['%d'%i for i in np.arange(5000,50000,5000)], cmap='coolwarm', ax=axs[2][1])
axs[2][1].set_title('Nt_pro_BNP')

'''results = proportional_hazard_test(cph, array, time_transform='rank')
results.print_summary(decimals=3, model="untransformed variables")
'''
#cph.check_assumptions(array, p_value_threshold=0.05)
cph.print_summary()
ahf = [float(array['AHF'][i]) for i in range(387)]
dur =[int(array['duration'][i]) for i in range(387)]
np.corrcoef(ahf, dur)
pearsonr(ahf, dur)
spearmanr(ahf, dur)
kendalltau(ahf, dur)


event_one_year = array['duration']<365
event_ten_year = array['duration']<3650
duration_one_year = pd.DataFrame(array['duration'])
duration_one_year[duration_one_year>365]=365
duration_ten_year=pd.DataFrame(array['duration'])
duration_ten_year[duration_ten_year>3650]=3650

event = [event_one_year, event_ten_year]
duration =[duration_one_year, duration_ten_year]
days=[0,2000]
title=['1-year Kaplan-Meier plots', '10-year Kaplan-Meier plots']
position = [0.75, 0.9]
for i in range(2):
    fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize=(20,20))
    km = KaplanMeierFitter()
    above = array['AHF']>np.median(array['AHF'].astype(float))
    km.fit(durations=duration[i][above], event_observed=event[i][above], label='High AHF score')
    km.plot(ax=axs[0][0])
    km.fit(durations=duration[i][~above], event_observed=event[i][~above], label='Low AHF score')
    km.plot(ax=axs[0][0])
    axs[0][0].set_title("AHF score")
    axs[0][0].set_xlabel("Survival time")
    axs[0][0].set_ylabel("Survival rate")
    log = logrank_test(duration[i][above], duration[i][~above], event_observed_A=event[i][above], event_observed_B=event[i][~above])
    axs[0][0].text(days[i], position[i], 'Log rank p-value: %.3f\nMedian = %.3f'%(log.p_value, np.median(array['AHF'].astype(float))))
    
    above = array['PAD']=='1.0'
    km.fit(durations=duration[i][above], event_observed=event[i][above], label='PAD')
    km.plot(ax=axs[0][1])
    km.fit(durations=duration[i][~above], event_observed=event[i][~above], label='No PAD')
    km.plot(ax=axs[0][1])
    axs[0][1].set_title("PAD")
    axs[0][1].set_xlabel("Survival time")
    axs[0][1].set_ylabel("Survival rate")
    log = logrank_test(duration[i][above], duration[i][~above], event_observed_A=event[i][above], event_observed_B=event[i][~above])
    axs[0][1].text(days[i], position[i], 'Log rank p-value: %.3f'%log.p_value)
    
    above = array['PCI']=='1.0'
    km.fit(durations=duration[i][above], event_observed=event[i][above], label='PCI')
    km.plot(ax=axs[1][0])
    km.fit(durations=duration[i][~above], event_observed=event[i][~above], label='No PCI')
    km.plot(ax=axs[1][0])
    axs[1][0].set_title("PCI")
    axs[1][0].set_xlabel("Survival time")
    axs[1][0].set_ylabel("Survival rate")
    log = logrank_test(duration[i][above], duration[i][~above], event_observed_A=event[i][above], event_observed_B=event[i][~above])
    axs[1][0].text(days[i], position[i], 'Log rank p-value: %.3f'%log.p_value)
    
    
    above = array['Age'].astype(float)>np.median(array['Age'].astype(float))
    km.fit(durations=duration[i][above], event_observed=event[i][above], label='The older')
    km.plot(ax=axs[1][1])
    km.fit(durations=duration[i][~above], event_observed=event[i][~above], label='The younger')
    km.plot(ax=axs[1][1])
    axs[1][1].set_title("Age")
    axs[1][1].set_xlabel("Survival time")
    axs[1][1].set_ylabel("Survival rate")
    log = logrank_test(duration[i][above], duration[i][~above], event_observed_A=event[i][above], event_observed_B=event[i][~above])
    axs[1][1].text(days[i], position[i], 'Log rank p-value: %.3f\nMedian = %.3f'%(log.p_value, np.median(array['Age'].astype(float))))
    
    
    above = array['BUN'].astype(float)>np.median(array['BUN'].astype(float))
    km.fit(durations=duration[i][above], event_observed=event[i][above], label='Higher BUN')
    km.plot(ax=axs[2][0])
    km.fit(durations=duration[i][~above], event_observed=event[i][~above], label='Lower BUN')
    km.plot(ax=axs[2][0])
    axs[2][0].set_title("BUN")
    axs[2][0].set_xlabel("Survival time")
    axs[2][0].set_ylabel("Survival rate")
    log = logrank_test(duration[i][above], duration[i][~above], event_observed_A=event[i][above], event_observed_B=event[i][~above])
    axs[2][0].text(days[i], position[i], 'Log rank p-value: %.3f\nMedian = %.3f'%(log.p_value, np.median(array['BUN'].astype(float))))
    
    
    above = array['Nt_pro_BNP'].astype(float)>np.median(array['Nt_pro_BNP'].astype(float))
    km.fit(durations=duration[i][above], event_observed=event[i][above], label='Higher NT-pro-BNP')
    km.plot(ax=axs[2][1])
    km.fit(durations=duration[i][~above], event_observed=event[i][~above], label='Lower NT-pro-BNP')
    km.plot(ax=axs[2][1])
    axs[2][1].set_title("NT-pro-BNP")
    axs[2][1].set_xlabel("Survival time")
    axs[2][1].set_ylabel("Survival rate")
    log = logrank_test(duration[i][above], duration[i][~above], event_observed_A=event[i][above], event_observed_B=event[i][~above])
    axs[2][1].text(days[i], position[i], 'Log rank p-value: %.3f\nMedian = %.3f'%(log.p_value, np.median(array['Nt_pro_BNP'].astype(float))))
    
    fig.suptitle('%s'%title[i], fontsize=20, fontweight="bold")
    fig.tight_layout()
    plt.show()

sort = array.iloc[:,[84,87]].sort_values('AHF', ascending=True)
roc = np.array([[0,0]])
for i in range(10):
    tpr = len(np.where(sort.iloc[int(len(sort)*i/10):int(len(sort)*(i+1)/10),1]<365)[0])/67
    fpr = len(np.where(sort.iloc[int(len(sort)*i/10):int(len(sort)*(i+1)/10),1]>365)[0])/320
    roc = np.append(roc, [[tpr, fpr]+roc[i]], axis=0)
    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(roc[:,0],roc[:,1], lw=2, label= "1-year survival time, area = %0.2f" % auc(roc[:,0],roc[:,1]))
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.legend(loc='lower right')
roc = np.array([[0,0]])
for i in range(10):
    tpr = len(np.where(sort.iloc[int(len(sort)*i/10):int(len(sort)*(i+1)/10),1]<1826)[0])/173
    fpr = len(np.where(sort.iloc[int(len(sort)*i/10):int(len(sort)*(i+1)/10),1]>1826)[0])/214
    roc = np.append(roc, [[tpr, fpr]+roc[i]], axis=0)
    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(roc[:,0],roc[:,1], lw=2, label= "5-year survival time, area = %0.2f" % auc(roc[:,0],roc[:,1]))
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.legend(loc='lower right')
roc = np.array([[0,0]])
for i in range(10):
    tpr = len(np.where(sort.iloc[int(len(sort)*i/10):int(len(sort)*(i+1)/10),1]<3653)[0])/294
    fpr = len(np.where(sort.iloc[int(len(sort)*i/10):int(len(sort)*(i+1)/10),1]>3653)[0])/93
    roc = np.append(roc, [[tpr, fpr]+roc[i]], axis=0)
    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(roc[:,0],roc[:,1], lw=2, label= "10-year survival time, area = %0.2f" % auc(roc[:,0],roc[:,1]))
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.legend(loc='lower right')




roc = np.array([[0,0]])
for i in range(10):
    tpr = np.sum((array['AHF']>0.1*i) & (array['AHF']<=0.1*(i+1)) & (array['duration']<365))/67
    fpr = np.sum((array['AHF']>0.1*i) & (array['AHF']<=0.1*(i+1)) & (array['duration']>365))/320
    roc = np.append(roc, [[tpr, fpr]+roc[i]], axis=0)
    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(roc[:,0],roc[:,1], lw=2, label= "1-year survival time, area = %0.2f" % auc(roc[:,0],roc[:,1]))
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.legend(loc='lower right')

roc = np.array([[0,0]])
for i in range(10):
    tpr = np.sum((array['AHF']>0.1*i) & (array['AHF']<=0.1*(i+1)) & (array['duration']<3653))/294
    fpr = np.sum((array['AHF']>0.1*i) & (array['AHF']<=0.1*(i+1)) & (array['duration']>3653))/93
    roc = np.append(roc, [[tpr, fpr]+roc[i]], axis=0)


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(roc[:,0],roc[:,1], lw=2, label= "10-year survival time, area = %0.2f" % auc(roc[:,0],roc[:,1]))
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.legend(loc='lower right')
