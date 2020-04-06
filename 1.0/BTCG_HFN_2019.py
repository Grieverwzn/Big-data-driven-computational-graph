# In[1]: import modules
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
#import multiprocessing
#cores = multiprocessing.cpu_count()
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
plt.rcParams['savefig.dpi']=100
plt.rcParams['figure.dpi']=100
import datetime
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[2] Input Data
class DNode:
    def __init__(self, node_id, is_zone, is_ozone):
        self.node_id = node_id
        self.is_zone = is_zone
        self.is_ozone = is_ozone

class DOzone:
    def __init__(self,node_id,ozone_id, target_generation, is_observed):
        self.node_id = node_id
        self.ozone_id = ozone_id
        self.target_generation = target_generation
        self.is_observed= is_observed
        
class DOD:
    def __init__(self, od_id, from_zone_id, to_zone_id, OD_split,is_observed):
        self.od_id = od_id
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.OD_split = OD_split
        self.is_observed= is_observed
        
class DPath:
    def __init__(self, path_id, from_zone_id, to_zone_id,K, node_sequence,link_sequence,
                 target_path_proportion,path_toll,path_travel_time, is_observed):
        self.path_id = path_id
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.K=K
        self.node_sequence = node_sequence
        self.link_sequence=link_sequence
        self.target_path_proportion = target_path_proportion
        self.path_toll=path_toll
        self.path_travel_time=path_travel_time
        self.is_observed= is_observed


class DLink:
    def __init__(self, link_id, from_node_id, to_node_id, link_pair, length,
                 sensor_name, observed_speed,observed_travel_time,
                 sensor_count,toll, capacity,is_observed):
        self.link_id = link_id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.link_pair=link_pair
        self.length = length
        self.sensor_name = sensor_name
        self.length=length
        self.observed_speed=observed_speed
        self.observed_travel_time = observed_travel_time
        self.sensor_count=sensor_count
        self.toll = toll
        self.capacity = capacity
        self.is_observed= is_observed
        
        
# In[3] Graph Class
class Ozone:
    def __init__(self, dozone,batch_size):
        self.node_id = dozone.node_id
        self.ozone_id = dozone.ozone_id
        self.target_generation=tf.placeholder(shape=[None,1],dtype=tf.float32, name='target_generation_'+str(self.ozone_id))
        self.initial_estimate_generation=np.mean(dozone.target_generation)*0.9 # Here set initial estimation
        self.estimate_generation = tf.Variable(100, dtype=tf.float32, name='estimate_generation_'+str(self.ozone_id))
        self.including_od=list()
        self.is_observed=dozone.is_observed
        #self.gamma_d = None # dominator

class OD:
    def __init__(self, dod,batch_size):
        self.od_id = dod.od_id
        self.from_zone_id = dod.from_zone_id
        self.to_zone_id = dod.to_zone_id
        self.OD_split = tf.placeholder(dtype=tf.float32,shape=[None,1],name='OD_split_'+str(dod.od_id))
        self.initial_gamma=np.mean(dod.OD_split) # Here set initial estimation
        #self.gamma_n = tf.Variable(0.1, dtype=tf.float32, name='gamma_'+str(dod.od_id)) # numerator
        self.gamma = tf.Variable(0.1, dtype=tf.float32, name='gamma_'+str(dod.od_id))
        self.estimate_OD_flow=tf.Variable(1,dtype=tf.float32, name='estimated_OD_flow_'+str(dod.od_id))
        self.theta_time=tf.Variable(0.1, dtype=tf.float32, name='theta_time_'+str(dod.od_id))
        self.theta_toll=tf.Variable(0.1, dtype=tf.float32, name='theta_toll_'+str(dod.od_id))
        self.theta_constant=tf.Variable(0.0, dtype=tf.float32, name='theta_constant'+str(dod.od_id))
        self.average_travel_time=None
        self.including_path =list()
        self.belonged_ozone=[]
        self.exp_reci_=None
        self.is_observed=dod.is_observed
         # for marginal analysis
        self.m_exp_reci_=None
        
class Path:
    def __init__(self, dpath,batch_size):
        self.path_id = dpath.path_id
        self.from_zone_id = dpath.from_zone_id
        self.to_zone_id = dpath.to_zone_id
        self.node_sequence = dpath.node_sequence
        self.link_sequence = dpath.link_sequence
        self.K=dpath.K
        self.path_toll=dpath.path_toll       
        self.initial_rou=np.mean(dpath.target_path_proportion)
        self.rou = tf.Variable(self.initial_rou, dtype=tf.float32, name='rou_'+str(dpath.path_id))       
        self.path_flow=tf.Variable(1, dtype=tf.float32, name='path_flow_'+str(dpath.path_id))       
        self.target_path_proportion=tf.placeholder(dtype=tf.float32,shape=[None,1], name='target_path_proportion_'+str(dpath.path_id))
        self.including_link=list()
        self.path_travel_time=dpath.path_travel_time
        self.belonged_od=[]
        self.exp_=None      
        self.is_observed=dpath.is_observed
        # for marginal analysis
        self.m_exp_=None 
        self.m_path_toll=None 
        self.m_path_flow=None
        self.m_rou=None 
        
class Link:
    def __init__(self,dlink,batch_size):
        self.link_id = dlink.link_id
        self.from_node_id = dlink.from_node_id
        self.to_node_id = dlink.to_node_id
        self.link_pair = dlink.link_pair
        self.toll = tf.constant(dlink.toll)
        self.capacity = tf.constant(dlink.capacity,tf.float32)
        self.observed_travel_time = dlink.observed_travel_time
        self.sensor_count = tf.placeholder(tf.float32,shape=[None,1], name='sensor_count_'+str(dlink.link_id))
        self.estimate_link_flow = None
        self.belonged_path = list()
        self.is_observed = dlink.is_observed


# In[1] function to load total samples
def load_data(): 
    
    t0 = datetime.datetime.now()
    node_df = pd.read_csv('input_node.csv',encoding='gbk')
    ozone_df = pd.read_csv('input_ozone.csv',encoding='gbk')
    survey_df=pd.read_csv('input_ozone_survey.csv',encoding='gbk')
    od_df = pd.read_csv('input_od.csv',encoding='gbk')
    mobile_df=pd.read_csv('input_od_mobile.csv',encoding='gbk')
    path_df = pd.read_csv('input_path.csv',encoding='gbk')
    float_df=pd.read_csv('input_path_float.csv',encoding='gbk')
    link_df = pd.read_csv('input_link.csv',encoding='gbk')
    sensor_df =pd.read_csv('input_link_sensor.csv', encoding='utf-8')
    t1 = datetime.datetime.now()
    print('Preparing Data using time',t1-t0,'\n')
            
            
            
    t0 = datetime.datetime.now()      
    # A: Build up auxilary attributes for node_df and ozone_df
    ozone_node_set = set(ozone_df.node_id)
    node_df['is_zone'] = node_df.apply(lambda x: True if x.zone_id!=-1 else False, axis=1)
    node_df['is_ozone'] = node_df.apply(lambda x: True if x.node_id in ozone_node_set else False, axis=1)
    survey_df['target_generation']=survey_df.apply(lambda x: x.trip_rate*x.population, axis=1)
    #ozone_df['is_observed']=ozone_df.apply(lambda x: True if x.trip_rate!=-1 or x.population = -1 else False, axis =1)
    
    # B: Build up auxilary attributes for od_df 
    prod_od_dict = mobile_df[['sample_id','from_zone_id','OD_demand']].groupby(['sample_id','from_zone_id']).sum()['OD_demand'] #小区出行生成量　using modbile phone data
    mobile_df['OD_split'] = mobile_df.apply(lambda x: x.OD_demand/prod_od_dict[x.sample_id,x.from_zone_id], axis=1)   #Calculate split_ratio using mobile phone data
    mobile_df['od_pair']=mobile_df.apply(lambda x: (x.from_zone_id,x.to_zone_id),axis=1)
    od_df['od_pair']=od_df.apply(lambda x: (x.from_zone_id,x.to_zone_id),axis=1)
    
    
    # C: Build up auxilary attributes for link_df
    link_df['link_pair'] = link_df.apply(lambda x: (x.from_node_id, x.to_node_id), axis=1) # name each link
    link_df['observed_travel_time']=link_df.apply(lambda x: x.length/x.observed_speed,axis=1) #
    link_df['is_observed']=link_df.apply(lambda x: True if x.sensor_name!='-1' else False, axis =1)
    
    # D: Build up auxilary variables for path_df
    path_link_dict={}
    path_toll_dict={}
    path_travel_time_dict={}
    link_toll_dict=link_df[['link_pair','toll']].set_index('link_pair').to_dict()['toll'] # in the dictionary each link only corresponds to one toll
    link_travel_time_dict=link_df[['link_pair','observed_travel_time']].set_index('link_pair').to_dict()['observed_travel_time'] # in the dictionary each link only corresponds to one toll
    

    for i in range(len(path_df)):
        path_r = path_df.loc[i]   #循环取出path_df中的每一行
        node_list = path_r.node_sequence.split(';')     #取出除了倒数第一个元素外其他所有元素，用；隔开
        link_list=list()
        path_toll=0
        path_travel_time=0
        for link_l in range(len(node_list)-1):
            link_pair=(int(node_list[len(node_list)-1-link_l]),int(node_list[len(node_list)-2-link_l]))
            #link_pair=(int(node_list[link_l]),int(node_list[link_l+1]))
            link_list.append(link_pair)
            path_toll=path_toll+link_toll_dict[link_pair]
            path_travel_time=path_travel_time+link_travel_time_dict[link_pair]
        path_link_dict[path_r.path_id]=link_list
        path_toll_dict[path_r.path_id]=path_toll
        path_travel_time_dict[path_r.path_id]=path_travel_time
    
    path_df['link_sequence']=path_df.apply(lambda x: path_link_dict[x.path_id],axis=1) # Calculate the link sequence
    path_df['path_toll']=path_df.apply(lambda x: path_toll_dict[x.path_id],axis=1) # Calculate the total toll
    path_df['path_travel_time']=path_df.apply(lambda x: path_travel_time_dict[x.path_id],axis=1) # Calculate the total toll
    path_df['path_tuple']=path_df.apply(lambda x: (x.from_zone_id,x.to_zone_id,x.K),axis=1)
    float_df['path_tuple']=float_df.apply(lambda x: (x.from_zone_id,x.to_zone_id,x.K),axis=1)
    
    t1 = datetime.datetime.now()
    print('Build up auxilary attributes for tables using time',t1-t0,'\n')
    
    return{'node_df':node_df, 'ozone_df':ozone_df, 'survey_df': survey_df,
           'od_df':od_df, 'mobile_df':mobile_df,
           'path_df':path_df, 'float_df':float_df,
           'link_df':link_df,'sensor_df':sensor_df,
           'nb_survey_sample': max(survey_df.sample_id),'nb_mobile_sample': max(mobile_df.sample_id),
           'nb_float_sample': max(float_df.sample_id),'nb_sensor_sample': max(sensor_df.sample_id)}


# In[2] function to load a certain sample
def initial(data): 
    # input total samples
    node_df = data['node_df']
    ozone_df = data['ozone_df']
    od_df = data['od_df']
    path_df = data['path_df']
    link_df = data['link_df']
    survey_df = data['survey_df']
    mobile_df = data['mobile_df']
    float_df = data['float_df']
    sensor_df=data['sensor_df']
    nb_survey_sample=data['nb_survey_sample']
    nb_mobile_sample=data['nb_mobile_sample']
    nb_float_sample=data['nb_float_sample']
    nb_sensor_sample=data['nb_sensor_sample']
    no_survey_list=np.zeros(nb_survey_sample)-1
    no_survey_list=no_survey_list.tolist()
    no_mobile_list=np.zeros(nb_mobile_sample)-1
    no_mobile_list=no_mobile_list.tolist()    
    no_float_list=np.zeros(nb_float_sample)-1
    no_float_list=no_float_list.tolist()
    no_sensor_list=np.zeros(nb_sensor_sample)-1
    no_sensor_list=no_sensor_list.tolist()
    # input the data of survey
    ozone_survey_dict={}
    for i in range(len(ozone_df)):
        ozone_o = ozone_df.loc[i]
        ozone_survey_list=list()
        if ozone_o.is_observed == True:
            #for j in range(len(survey_df)):
            for j in range(nb_survey_sample):
                #survey_s=survey_df.loc[j]
                survey_s=survey_df.loc[j*len(ozone_df)+i]
                ozone_survey_list.append(survey_s.target_generation)
    #            if ozone_o.ozone_id==survey_s.ozone_id:
    #                ozone_survey_list.append(survey_s.target_generation)
        ozone_survey_dict[ozone_o.ozone_id]=ozone_survey_list
    
    ozone_df['target_generation']=ozone_df.apply(lambda x: ozone_survey_dict[x.ozone_id] if x.is_observed==True else no_survey_list, axis=1)
    
    
    # Input the data of mobile phone
    od_mobile_dict={}
    for i in range(len(od_df)):
        od_w = od_df.loc[i]
        od_mobile_list=list()
        if od_w.is_observed==True:
            #for j in range(len(mobile_df)):
            for j in range(nb_mobile_sample):
                mobile_m=mobile_df.loc[j*len(od_df)+i]
                od_mobile_list.append(mobile_m.OD_split)
                #if od_w.od_pair==mobile_m.od_pair:
                #    od_mobile_list.append(mobile_m.OD_split)
        od_mobile_dict[od_w.od_pair]=od_mobile_list   

    od_df['OD_split']=od_df.apply(lambda x: od_mobile_dict[x.od_pair] if x.is_observed==True else no_mobile_list, axis=1)
     
     # Input the data of float car data
    path_float_dict={}
    for i in range(len(path_df)):
        path_p = path_df.loc[i]
        path_float_list=list()
        if path_p.is_observed==True:
            #for j in range(len(float_df)):
            for j in range(nb_float_sample):
                #float_f=float_df.loc[j]
                float_f=float_df.loc[j*len(path_df)+i]
                path_float_list.append(float_f.target_path_proportion)
                #if path_p.path_tuple==float_f.path_tuple:
                #    path_float_list.append(float_f.target_path_proportion)
        path_float_dict[path_p.path_tuple]=path_float_list
    path_df['target_path_proportion']=path_df.apply(lambda x: path_float_dict[x.path_tuple] if x.is_observed==True else no_float_list, axis=1)
    
     # Input the data of sensor data
    link_sensor_dict = link_df[['link_pair','sensor_name']].set_index('link_pair').to_dict()['sensor_name'] #sensor id 和link_id的对应关系
    link_count_dict={}
    for l in range(len(link_df)):
        link_l = link_df.loc[l]
        link_count_list=list()
        if link_l.is_observed==True:
            for s in range(len(sensor_df)):
                sensor_s=sensor_df.loc[s]
                if link_l.sensor_name==sensor_s.sensor_name:
                    link_count_list.append(sensor_s.sensor_count)
        link_count_dict[link_l.link_pair]=link_count_list
          
    link_df['sensor_count']=link_df.apply(lambda x: link_count_dict[x.link_pair] if link_sensor_dict[x.link_pair]!='-1' else no_sensor_list, axis=1)
    


    # Input data to the data classes
    dnode = node_df.apply(lambda x: DNode(node_id=x.node_id,
                                            is_zone=x.is_zone,
                                            is_ozone=x.is_ozone),axis=1)
    
  
    dozone=ozone_df.apply(lambda x: DOzone(ozone_id=x.ozone_id, 
                                          node_id=x.node_id,
                                          target_generation=x.target_generation,
                                          is_observed=x.is_observed),axis=1)
    
    dod=od_df.apply(lambda x:DOD(od_id=x.od_id,
                                     from_zone_id=x.from_zone_id,
                                     to_zone_id=x.to_zone_id,
                                     OD_split = x.OD_split,
                                     is_observed=x.is_observed),axis=1)
    
    dpath = path_df.apply(lambda x: DPath(path_id=x.path_id,
                                          from_zone_id=x.from_zone_id,
                                          to_zone_id=x.to_zone_id,
                                          K=x.K,
                                          node_sequence=list(map(lambda y: int(y),x.node_sequence.split(';'))),
                                          link_sequence=x.link_sequence,
                                          target_path_proportion=x.target_path_proportion,
                                          path_toll=x.path_toll,
                                          path_travel_time=x.path_travel_time,
                                          is_observed=x.is_observed), axis=1)
                                         
    dlink=link_df.apply(lambda x: DLink(link_id=x.link_id,
                                        from_node_id=x.from_node_id,
                                        to_node_id=x.to_node_id,
                                        link_pair=x.link_pair,
                                        length=x.length,
                                        observed_speed=x.observed_speed,
                                        toll=x.toll,
                                        capacity=x.capacity,
                                        sensor_name=x.sensor_name,
                                        observed_travel_time=x.observed_travel_time,
                                        sensor_count=x.sensor_count,
                                        is_observed=x.is_observed
                                        ),axis=1)
    return{'dozone':dozone, 'dnode':dnode, 'dod':dod, 'dpath':dpath, 'dlink':dlink}


# In[4] Building loss
def build_loss(ozone_set, od_set, path_set, link_set):
    # Constraints of TFDE-UB
    # 'Building Connections from Ozone layer to OD layer'
    t0 = datetime.datetime.now()
    for od in od_set:
        for ozone in ozone_set:
            if od.from_zone_id==ozone.ozone_id:
                od.belonged_ozone=ozone
                ozone.including_od.append(od)
#                od.estimate_OD_flow=tf.multiply(od.gamma, ozone.estimate_generation)
#   
#    for od in od_set:
#        od.estimate_OD_flow=tf.multiply(od.gamma, od.belonged_ozone.estimate_generation)
        
    for path in path_set:
        for od in od_set:
            if od.from_zone_id == path.from_zone_id and od.to_zone_id == path.to_zone_id:
                path.belonged_od = od
                od.including_path.append(path)           

#    for path in path_set:
#        path.exp_ = tf.exp(-path.belonged_od.theta_time * path.path_travel_time - path.belonged_od.theta_toll * path.path_toll+ path.belonged_od.theta_constant)
#
#    for od in od_set:
#        od.exp_reci_ = tf.reciprocal(tf.reduce_sum(tf.stack([path.exp_ for path in od.including_path])))
#    
#    for od in od_set:
#        od.average_travel_time=np.mean(np.stack([path.path_travel_time for path in od.including_path],axis=0))
#        for path in od.including_path:
#            path.rou = tf.multiply(path.exp_, od.exp_reci_)
#            path.path_flow = tf.multiply(path.rou, od.estimate_OD_flow)
#    
    for ozone in ozone_set:
        #ozone.gamma_d=tf.reciprocal(tf.reduce_sum(tf.stack([od.gamma_n for od in ozone.including_od])))
        for od in ozone.including_od:
             #od.gamma=tf.multiply(od.gamma_n, ozone.gamma_d)
             od.estimate_OD_flow=tf.maximum(tf.multiply(od.gamma, ozone.estimate_generation),0.000001)
             #od.estimate_OD_flow=tf.multiply(od.gamma, od.belonged_ozone.estimate_generation)
             for path in od.including_path:
                path.exp_ = tf.exp(-path.belonged_od.theta_time * path.path_travel_time - path.belonged_od.theta_toll * path.path_toll+ path.belonged_od.theta_constant)             
             od.exp_reci_ = tf.reciprocal(tf.reduce_sum(tf.stack([path.exp_ for path in od.including_path])))
             #od.average_travel_time=np.mean(np.stack([path.path_travel_time for path in od.including_path],axis=0))
             for path in od.including_path:
                 path.rou = tf.multiply(path.exp_, od.exp_reci_)
                 path.path_flow = tf.maximum(tf.multiply(path.rou, od.estimate_OD_flow),0.000001)
                 #path.path_flow = tf.multiply(path.rou, od.estimate_OD_flow)
                

#    # 'Building Connections from OD layer to path layer'   
#    for path in path_set:
#        path.rou = tf.multiply(path.exp_, path.belonged_od.exp_reci_)
#        path.path_flow = tf.multiply(path.rou, path.belonged_od.estimate_OD_flow)
    
    # 'Building Connections from Logit layer to path layer'
    link_dict = dict()
    for link in link_set:
        link_dict[(link.from_node_id, link.to_node_id)] = link
        
    for path in path_set:
        link_list = path.link_sequence
        for i in range(1,len(link_list)+1):
            link = link_dict[link_list[i-1]]
            link.belonged_path.append(path)            
    # 'Building Connections from Path layer to link layer' 
    for link in link_set:
        link.estimate_link_flow = tf.reduce_sum(tf.stack([path.path_flow for path in link.belonged_path]))


    
    f_survey = tf.reduce_sum(tf.stack([tf.pow(tf.subtract(tf.truediv(ozone.estimate_generation, ozone.target_generation),1),2) for ozone in ozone_set if ozone.is_observed]),name='loss/f_survey_sum') 
    tf.summary.scalar('loss/f_survey_sum', f_survey)
    
    f_mobile = tf.reduce_sum(tf.stack([tf.pow(tf.subtract(tf.truediv(od.gamma,od.OD_split),1),2) for od in od_set if od.is_observed]),name='loss/f_mobile_sum')
    #f_mobile = tf.reduce_sum(tf.stack([tf.pow(tf.subtract(od.gamma,od.OD_split),2) for od in od_set if od.is_observed]),name='loss/f_mobile_sum')
    tf.summary.scalar('loss/f_mobile_sum', f_mobile)
    
    f_float = tf.reduce_sum(tf.stack([tf.pow(tf.subtract(tf.truediv(path.rou,path.target_path_proportion),1),2) for path in path_set if path.is_observed]),name='loss/f_float_sum')
    tf.summary.scalar('loss/f_float_sum', f_float)
    
    f_count = tf.reduce_sum(tf.stack([tf.pow(tf.subtract(tf.truediv(link.estimate_link_flow,link.sensor_count),1),2) for link in link_set if link.is_observed]),name='loss/f_count_sum')
    tf.summary.scalar('loss/f_count_sum', f_count)

    #loss = tf.reduce_sum(tf.stack([f_survey, f_mobile, f_float,f_count]),name='loss/total_sum')
    loss = tf.reduce_sum(tf.stack([f_survey, f_mobile, f_float,f_count]),name='loss/total_sum')
    #loss = tf.reduce_sum([f_survey, f_mobile, f_float,f_count],name='loss/total_sum')
    tf.summary.scalar('loss/total_sum', loss)
    
    t1 = datetime.datetime.now()
    print('Building loss using Time: ',t1-t0,'\n') 

    return f_survey, f_mobile, f_float, f_count,loss
   # return f_survey, f_mobile, f_count

# In[5] Building optimizer
def build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile,f_float,lr_float, f_count, lr_count,loss, learning_rate):
#def build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile,f_count, lr_count):
    t0 = datetime.datetime.now()
    my_opt_survey=tf.train.GradientDescentOptimizer(lr_survey)
    #my_opt_survey=tf.train.AdagradOptimizer(lr_survey)
    opt_survey = my_opt_survey.minimize(f_survey)
    t1 = datetime.datetime.now()
    print('Building Optimizer survey using Time: ',t1-t0,'\n')
    
    my_opt_mobile=tf.train.GradientDescentOptimizer(lr_mobile)
    #my_opt_mobile=tf.train.AdagradOptimizer(lr_mobile)
    opt_mobile = my_opt_mobile.minimize(f_mobile)
    t2 = datetime.datetime.now()
    print('Building Optimizer mobile using Time: ',t2-t0,'\n')

    my_opt_float=tf.train.GradientDescentOptimizer(lr_float)
    #my_opt_float=tf.train.AdagradOptimizer(lr_float)
    opt_float = my_opt_float.minimize(f_float)
    t2 = datetime.datetime.now()
    print('Building Optimizer float using Time: ',t2-t0,'\n')
        
    my_opt_count=tf.train.GradientDescentOptimizer(lr_count)
    #my_opt_count=tf.train.AdagradOptimizer(lr_count)
    opt_count=my_opt_count.minimize(f_count)
    t3 = datetime.datetime.now()
    print('Building Optimizer sensor using Time: ',t3-t0,'\n')
     
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    opt_total=my_opt.minimize(loss)
    
    
    print('Adding moniter...')
    alpha_ = tf.stack([ozone.estimate_generation for ozone in ozone_set], name='alpha/vector')
    tf.summary.histogram('alpha/vector', alpha_) ####
    
    gamma_ = tf.stack([od.gamma for od in od_set], name='gamma/vector')
    tf.summary.histogram('gamma/vector', gamma_) ####
    
    VOT=tf.stack([od.theta_toll/od.theta_time for od in od_set], name='VOT/vector')
    tf.summary.histogram('VOT/vector',VOT)
    
    theta_constant_=tf.stack([od.theta_constant for od in od_set], name='theta_constant/vector')
    tf.summary.histogram('theta_constant/vector',theta_constant_)
    
    q_ = tf.stack([od.estimate_OD_flow for od in od_set], name='q/vector')
    tf.summary.histogram('q/vector', q_) ####
    
    rou_ = tf.stack([path.rou for path in path_set], name='rou/vector')
    tf.summary.histogram('rou/vector', rou_) ####
    
    flow_ = tf.stack([path.path_flow for path in path_set], name='flow/vector')
    tf.summary.histogram('flow/vector', flow_) ####
    
    v_ = tf.stack([link.estimate_link_flow for link in link_set], name='v/vector')
    tf.summary.histogram('v/vector', v_ ) ####


    
    return opt_survey, opt_mobile, opt_float, opt_count,opt_total
    #return opt_survey, opt_mobile, opt_count

# In[3] Feed
def feed_data(data, graph,rand_survey_index,rand_mobile_index,rand_float_index,rand_sensor_index,batch_size):
    dozone = data['dozone']
    dod = data['dod']
    dpath = data['dpath']
    dlink = data['dlink']
    feed = dict()
#    t0 = datetime.datetime.now()
    
    for ozone in dozone:
            generation_0=np.array(ozone.target_generation,dtype=np.float)
            generation_1=generation_0[rand_survey_index]
            generation_name = graph.get_tensor_by_name('target_generation_'+str(ozone.ozone_id)+':0')
            generation=generation_1.reshape(batch_size,1)
            feed[generation_name] = generation

#    t1 = datetime.datetime.now()
#    print('ramdom sample survey data using Time: ',t1-t0,'\n')
    
    for od in dod:
            split_0=np.array(od.OD_split,dtype=np.float32)
            split_1=split_0[rand_mobile_index]
            split_name = graph.get_tensor_by_name('OD_split_'+str(od.od_id)+':0')
            split=split_1.reshape([batch_size,1])
            feed[split_name] = split
    
#    t2 = datetime.datetime.now()
#    print('ramdomly sample mobile data using Time: ',t2-t1,'\n')
    
    for path in dpath:
            proportion_0=np.array(path.target_path_proportion,dtype=np.float32)
            proportion_1=proportion_0[rand_float_index]
            proportion_name = graph.get_tensor_by_name('target_path_proportion_'+str(path.path_id)+':0')        
            proportion=proportion_1.reshape([batch_size,1])
            feed[proportion_name] = proportion.reshape([batch_size,1])
    
#    t3 = datetime.datetime.now()
#    print('ramdomly sample float data using time: ',t3-t2,'\n')
    
    for link in dlink:
            count_0=np.array(link.sensor_count,dtype=np.float32)
            count_1=count_0[rand_sensor_index]
            count_name = graph.get_tensor_by_name('sensor_count_'+str(link.link_id)+':0')
            count=count_1.reshape([batch_size,1])
            feed[count_name] = count.reshape([batch_size,1])
        
#    t4 = datetime.datetime.now()
#    print('ramdomly sample sensor data using Time: ',t4-t3,'\n')


    return feed

# In[3] output
def output_results(sess,data,feed,graph):
    est_alpha = sess.run(graph.get_tensor_by_name('alpha/vector:0'), feed_dict=feed)
    df_dict = {'ozone_id': [ozone.ozone_id for ozone in data['dozone']],
                     'estimate_generation': est_alpha,
                     'target_generation':[np.mean(ozone.target_generation) for ozone in data['dozone']]}
    ozone_df = pd.DataFrame(df_dict)
    ozone_df.to_csv('output_ozone.csv', index=None)

    est_gamma = sess.run(graph.get_tensor_by_name('gamma/vector:0'), feed_dict=feed)
    est_q=sess.run(graph.get_tensor_by_name('q/vector:0'),feed_dict=feed)
    est_vot = sess.run(graph.get_tensor_by_name('VOT/vector:0'), feed_dict=feed)
    est_theta_constant = sess.run(graph.get_tensor_by_name('theta_constant/vector:0'), feed_dict=feed)
    df_dict = {'from_zone_id': [od.from_zone_id for od in data['dod']],
                     'to_zone_id': [od.to_zone_id for od in data['dod']],
                     'estimate_gamma': est_gamma,
                     'estimate_OD_flow': est_q,
                     'target_OD_split': [np.mean(od.OD_split) for od in data['dod']],
                     'VOT': est_vot,
                     'theta_constant':est_theta_constant}
    od_df = pd.DataFrame(df_dict)
    od_df.to_csv('output_od.csv', index=None)
   
    est_rou = sess.run(graph.get_tensor_by_name('rou/vector:0'), feed_dict=feed)
    est_flow=sess.run(graph.get_tensor_by_name('flow/vector:0'),feed_dict=feed)
    df_dict = {'from_zone_id': [path_r.from_zone_id for path_r in data['dpath']],
               'to_zone_id': [path_r.to_zone_id for path_r in data['dpath']],
               'K': [path_r.K for path_r in data['dpath']],
               'node_sequence':[path_r.node_sequence for path_r in data['dpath']],
               'estimate_path_proportion': est_rou,
               'estimate_path_flow': est_flow,
               'target_proportion': [np.mean(path_r.target_path_proportion) for path_r in data['dpath']]}
    path_df = pd.DataFrame(df_dict)
    path_df.to_csv('output_path.csv', index=None)
    
    est_ = sess.run(graph.get_tensor_by_name('v/vector:0'), feed_dict=feed)
    df_dict = {'link_id': [link_l.link_id for link_l in data['dlink']],
                   'from_node_id': [link_l.from_node_id for link_l in data['dlink']],
                   'to_node_id': [link_l.to_node_id for link_l in data['dlink']],
                   'estimated_count': est_,
                   'target_count': [np.mean(link_l.sensor_count) for link_l in data['dlink']]}
    link_df = pd.DataFrame(df_dict)
    link_df.to_csv('output_link.csv', index=None)
    
# In[3] Logit model for marginal analysis
def m_logit(od_set,path_set,link_set,link,feed):
    
#    for path in path_set:
#        path.m_path_toll=path.path_toll
#        if path in link.belonged_path:path.m_path_toll=path.path_toll+1
#        path.m_exp_=np.exp(-sess.run(path.belonged_od.theta_time,feed_dict=feed) * path.path_travel_time - sess.run(path.belonged_od.theta_toll,feed_dict=feed) * path.m_path_toll+ sess.run(path.belonged_od.theta_constant,feed_dict=feed))       
    
    for od in od_set:
        for path in od.including_path:
            path.m_path_toll=path.path_toll
            if path in link.belonged_path:path.m_path_toll=path.path_toll+1.0
            path.m_exp_=np.exp(-sess.run(path.belonged_od.theta_time,feed_dict=feed) * path.path_travel_time - sess.run(path.belonged_od.theta_toll,feed_dict=feed) * path.m_path_toll+ sess.run(path.belonged_od.theta_constant,feed_dict=feed))       
        od.m_exp_reci_ = np.reciprocal(np.sum(np.stack([path.m_exp_ for path in od.including_path])))
        for path in od.including_path:
            path.m_rou=np.multiply(path.m_exp_, od.m_exp_reci_)
            path.m_path_flow = np.maximum(0.000001,np.multiply(path.m_rou, sess.run(path.belonged_od.estimate_OD_flow,feed_dict=feed)))
   
    m_estimate_link_flow_list=list()
    for link_l in link_set:
        m_estimate_link_flow = np.sum([path.m_path_flow for path in link_l.belonged_path])-sess.run(link_l.estimate_link_flow,feed_dict=feed) 
        m_estimate_link_flow_list.append(m_estimate_link_flow)
        
    #link.m_estimate_link_flow=m_estimate_link_flow_list
    
    return m_estimate_link_flow_list

# In[3] Marginal analysis
def margin_price(link_set, path_set,od_set,feed,bottleneck):
    margin_price={}
    margin_avg_price={}
    nb_link=0
    for link in link_set:
        if sess.run(link.estimate_link_flow,feed_dict=feed)>=bottleneck:
            nb_link=nb_link+1
            if link.link_id==39:
                A=1
            m_estimate_link_flow=m_logit(od_set,path_set,link_set,link,feed)
            margin_price[str(link.link_pair)]=pd.Series(m_estimate_link_flow)
            margin_avg_price[str(link.link_pair)]=np.mean(m_estimate_link_flow)
            plt.tight_layout()
            plt.bar(range(1,len(m_estimate_link_flow)+1),m_estimate_link_flow,fc='g')
            plt.xticks([x for x in range(1,len(m_estimate_link_flow)+1) if x%3==0 or x==1],fontsize=7)
            plt.title('Marginal effects of one extra toll on link'+str(link.link_id)+'/ link'+str(link.link_pair))
            plt.savefig('margin_price'+str(link.link_id)+str(link.link_pair)+'.svg',dpi=300, format="svg")
            plt.show()
    
    margin_price_df=pd.DataFrame.from_dict(margin_price)
    margin_price_df.to_csv('output_margin_price.csv', index=1)
    return margin_price_df,margin_avg_price

def margin_od(link_set,path_set,od_set,feed):
    t0 = datetime.datetime.now()
    margin_link_od={}
    for link in link_set:
        margin_link_od_list=list()
        margin_od_name_list=list()
        for od in od_set:
            m_od=0.0
            for path in od.including_path:
                if path in link.belonged_path: 
                    margin_od=sess.run(path.rou,feed_dict=feed)
                    m_od=m_od-margin_od
            margin_link_od_list.append(m_od)
            margin_od_name_list.append(str((od.from_zone_id,od.to_zone_id)))
        margin_link_od[str(link.link_pair)]=pd.Series(margin_link_od_list,index=margin_od_name_list)           
    
    margin_od_df=pd.DataFrame.from_dict(margin_link_od)
    margin_od_df.to_csv('output_margin_od.csv', index=1)
    
    return margin_od_df
    t1 = datetime.datetime.now()
    print('Margin calculating using Time: ',t1-t0,'\n')

def margin_ozone(link_set,path_set,od_set,ozone_set,feed):
    t0 = datetime.datetime.now()
    margin_link_ozone={}
    for link in link_set:
        margin_link_ozone_list=list()
        margin_ozone_name_list=list()
        for ozone in ozone_set:
            m_ozone=0.0
            for od in ozone.including_od:
                for path in od.including_path:
                    if path in link.belonged_path: 
                        margin_ozone=sess.run(path.rou*od.gamma,feed_dict=feed)
                        m_ozone=m_ozone-margin_ozone
            margin_link_ozone_list.append(m_ozone)
            margin_ozone_name_list.append(str(ozone.ozone_id))
        margin_link_ozone[str(link.link_pair)]=pd.Series(margin_link_ozone_list,index=margin_ozone_name_list)           
    
    margin_ozone_df=pd.DataFrame.from_dict(margin_link_ozone)
    margin_ozone_df.to_csv('output_margin_ozone.csv', index=1)
    
    return margin_ozone_df
    t1 = datetime.datetime.now()
    print('Margin calculating (ozone) using Time: ',t1-t0,'\n')
# In[3] Congestion component    
def congestion_component(link_set,path_set,od_set,ozone_set,feed,bottleneck):
    # Congestion component
    link_path_name_dict={}
    link_path_flow_dict={}
    link_od_name_dict={}
    link_od_flow_dict={}
    link_ozone_name_dict={}
    link_ozone_flow_dict={}
    link_path_margin_dict={}
    link_od_margin_dict={}
    link_ozone_margin_dict={}
    for link in link_set:
        od_scan_list=list()
        ozone_scan_list=list()
        if sess.run(link.estimate_link_flow,feed_dict=feed)>=bottleneck:
            belonged_path_flow_list=list()
            belonged_path_name_list=list()
            belonged_od_flow_list=list()
            belonged_od_name_list=list()
            belonged_ozone_flow_list=list()
            belonged_ozone_name_list=list()           
            for path in link.belonged_path:
                belonged_path_flow_list.append(sess.run(path.path_flow,feed_dict=feed))
                belonged_path_name_list.append(((path.from_zone_id,path.to_zone_id),path.K))
                if path.belonged_od not in od_scan_list:
                    od=path.belonged_od
                    KK=sess.run(tf.reduce_sum(tf.stack([path_r.path_flow for path_r in path.belonged_od.including_path if path_r in link.belonged_path])),feed_dict=feed)
                    belonged_od_flow_list.append(KK)
                    belonged_od_name_list.append((od.from_zone_id,od.to_zone_id))
                    od_scan_list.append(od)
                if path.belonged_od.belonged_ozone not in ozone_scan_list:
                    ozone=path.belonged_od.belonged_ozone
                    KK=0.0
                    for od_w in path.belonged_od.belonged_ozone.including_od:
                        KK=KK+sess.run(tf.reduce_sum(tf.stack([path_r.path_flow for path_r in od_w.including_path if path_r in link.belonged_path])),feed_dict=feed)
                    belonged_ozone_flow_list.append(KK)
                    belonged_ozone_name_list.append(ozone.ozone_id)
                    ozone_scan_list.append(ozone)                  

            link_path_flow_dict[link.link_pair]=belonged_path_flow_list
            link_path_name_dict[link.link_pair]=belonged_path_name_list 
            link_path_margin_dict[str(link.link_pair)]=pd.Series(belonged_path_flow_list,index=belonged_path_name_list)           
            link_path_margin_df=pd.DataFrame.from_dict(link_path_margin_dict)
            link_path_margin_df.to_csv('link_path_margin.csv', index=1)
            
            link_od_flow_dict[link.link_pair]=belonged_od_flow_list
            link_od_name_dict[link.link_pair]=belonged_od_name_list
            link_od_margin_dict[str(link.link_pair)]=pd.Series(belonged_od_flow_list, index=belonged_od_name_list)
            link_od_margin_df=pd.DataFrame.from_dict(link_od_margin_dict)
            link_od_margin_df.to_csv('link_od_margin_df.csv', index=1)
            
            link_ozone_flow_dict[link.link_pair]=belonged_ozone_flow_list
            link_ozone_name_dict[link.link_pair]=belonged_ozone_name_list          
            link_ozone_margin_dict[str(link.link_pair)]=pd.Series(belonged_ozone_flow_list, index=belonged_ozone_name_list)
            link_ozone_margin_df=pd.DataFrame.from_dict(link_ozone_margin_dict)
            link_ozone_margin_df.to_csv('link_ozone_margin_df.csv', index=1)
            plt.tight_layout()
#            patches,l_text,p_text =plt.pie(link_path_flow_dict[link.link_pair], labels=link_path_name_dict[link.link_pair], 
#                    autopct='%1.0f%%',
#                    shadow=False,labeldistance = 1.2,
#                    startangle = 90, pctdistance = 0.8)
            patches,l_text,p_text =plt.pie(link_path_flow_dict[link.link_pair], 
            autopct='%1.0f%%',
            shadow=False,labeldistance = 1.2,
            pctdistance = 0.8)
            plt.legend(labels=link_path_name_dict[link.link_pair],loc='upper left', bbox_to_anchor=(0.8,1.1))
            for t in l_text:
                t.set_size=('xx-small')
            for t in p_text:
                t.set_size=('xx-small')
            # 设置x，y轴刻度一致，这样饼图才能是圆的
            plt.axis('equal')
            #plt.legend()
            plt.savefig('link_path'+str(link.link_pair)+'.svg',dpi=300, format='svg')
            plt.show()
            
            plt.tight_layout()
#            patches,l_text,p_text =plt.pie(link_od_flow_dict[link.link_pair], labels=link_od_name_dict[link.link_pair], 
#                    autopct='%1.0f%%',
#                    shadow=False,labeldistance = 1.3,
#                    startangle = 90, pctdistance = 0.8)
            patches,l_text,p_text =plt.pie(link_od_flow_dict[link.link_pair], 
            autopct='%1.0f%%',
            shadow=False,labeldistance = 1.3,
            pctdistance = 0.8)
            plt.legend(labels=link_od_name_dict[link.link_pair],loc='upper left', bbox_to_anchor=(0.8,1.1))
            for t in l_text:
                t.set_size=('xx-small')
            for t in p_text:
                t.set_size=('xx-small')
            # 设置x，y轴刻度一致，这样饼图才能是圆的
            plt.axis('equal')
            plt.savefig('link_od'+str(link.link_pair)+'.svg',dpi=300, format='svg')
            plt.show()
            
            plt.tight_layout()
#            patches,l_text,p_text =plt.pie(link_ozone_flow_dict[link.link_pair], labels=link_ozone_name_dict[link.link_pair], 
#                    autopct='%1.0f%%',
#                    shadow=False,labeldistance = 1.3,
#                    startangle = 90, pctdistance = 0.8)
            patches,l_text,p_text =plt.pie(link_ozone_flow_dict[link.link_pair], 
                    autopct='%1.0f%%',
                    shadow=False,labeldistance = 1.3,
                    pctdistance = 0.8)       
            plt.legend(labels=link_ozone_name_dict[link.link_pair],loc='upper left', bbox_to_anchor=(0.8,1.1))
            for t in l_text:
                t.set_size=('xx-small')
            for t in p_text:
                t.set_size=('xx-small')
            # 设置x，y轴刻度一致，这样饼图才能是圆的
            plt.axis('equal')
            plt.savefig('link_ozone'+str(link.link_pair)+'.svg',dpi=300, format='svg')
            plt.show()            

#return link_ozone_margin_df,link_od_margin_df,link_path_margin_df
#labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
 #autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数 
 #shadow，饼是否有阴影
 #startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看 
 #pctdistance，百分比的text离圆心的距离 
 #patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本 


#plt.pie(link_path_flow_dict[(1,3)], labels=link_path_name_dict[(1,3)], autopct='%1.1f%%', shadow=True)
 
# In[1] Main loop
maximum_iterations=200;
curr_iter = tf.Variable(0,name='utils/iterator') ##############

#lr_survey=1000.0
#lr_mobile=0.01
#lr_float=0.1
#lr_count=0.001
#learning_rate=0.001


lr_survey = tf.train.exponential_decay(10000.0, curr_iter, decay_steps=maximum_iterations, decay_rate=0.5)
lr_mobile = tf.train.exponential_decay(0.003, curr_iter, decay_steps=maximum_iterations/5, decay_rate=0.5)
lr_float = tf.train.exponential_decay(0.1, curr_iter, decay_steps=maximum_iterations, decay_rate=0.5)
lr_count = tf.train.exponential_decay(0.002, curr_iter, decay_steps=maximum_iterations, decay_rate=0.5)
learning_rate = tf.train.exponential_decay(0.001, curr_iter, decay_steps=maximum_iterations, decay_rate=0.5)

bottleneck=400

# In[2] Loading data
t0 = datetime.datetime.now()
data_all = load_data()
t1 = datetime.datetime.now()
print('loading data using Time: ',t1-t0,'\n')   

nb_survey_sample=data_all['nb_survey_sample']
nb_mobile_sample=data_all['nb_mobile_sample']
nb_float_sample=data_all['nb_float_sample']
nb_sensor_sample=data_all['nb_sensor_sample']

batch_size=1
# In[3] Initialization
t0 = datetime.datetime.now()
data= initial(data_all)
t1 = datetime.datetime.now()
print('Initialization using Time: ',t1-t0,'\n')  


# In[4] Construct graph class
t0 = datetime.datetime.now()
ozone_set = [Ozone(dozone,batch_size) for dozone in data['dozone']]
od_set = [OD(dod,batch_size) for dod in data['dod']]
path_set = [Path(dpath,batch_size) for dpath in data['dpath']]
link_set = [Link(dlink,batch_size) for dlink in data['dlink']]
t1 = datetime.datetime.now()
print('Build up graph class using Time: ',t1-t0,'\n')



# In[4] Build computational graph
f_survey, f_mobile,f_float,f_count,loss = build_loss(ozone_set, od_set, path_set, link_set)
opt_survey, opt_mobile, opt_float,opt_count,opt_total = build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile, f_float, lr_float, f_count, lr_count,loss,learning_rate)

#f_survey, f_mobile,f_count = build_loss(ozone_set, od_set, path_set, link_set)
#opt_survey, opt_mobile,opt_count = build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile, f_count, lr_count)

# In[5] feeddata

#config =tf.ConfigProto(log_device_placement=True)

with tf.Session() as sess:
    # (1) Set tensorboard
    writer = tf.summary.FileWriter("./logs", sess.graph) ####
    merged = tf.summary.merge_all() #### Package computational graph
    # (2) Variable initialization
    t0 = datetime.datetime.now()
    
    init=tf.global_variables_initializer()
    sess.run(init)
    t1 = datetime.datetime.now()
    print('Initialize base data using Time: ',t1-t0,'\n')
    
    # (3) Get computational graph
    graph = tf.get_default_graph()
    t2 = datetime.datetime.now()
    print('Get graph using Time: ',t2-t1,'\n')

    list_survey=[]
    list_mobile=[]
    list_float=[]
    list_sensor=[]
    list_total=[]

    t0 = datetime.datetime.now()
    for nb_iter in range(maximum_iterations):
        t0 = datetime.datetime.now()
        # (1) Select a sample randomly
        rand_survey_index=np.random.choice(nb_survey_sample,batch_size)
        rand_mobile_index=np.random.choice(nb_mobile_sample,batch_size)
        rand_float_index=np.random.choice(nb_float_sample,batch_size)
        rand_sensor_index=np.random.choice(nb_sensor_sample,batch_size)
        # (2) Feed the data
        feed = feed_data(data, graph,rand_survey_index,rand_mobile_index,rand_float_index,rand_sensor_index,batch_size)
        # (3) Setting tensorboard
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) ####
        run_metadata = tf.RunMetadata() ####
        # 1) trace each iteration, e.g. tensorboard > graphs > session runs; 
        # 2) metadata also stores information like run times, memory consumption, e.g. 
        train_survey, _ = sess.run([f_survey, opt_survey], feed_dict=feed) ####
        train_mobile, _ = sess.run([f_mobile, opt_mobile], feed_dict=feed)
        train_float, _ = sess.run([f_float, opt_float], feed_dict=feed)
        train_sensor, _ = sess.run([f_count, opt_count], feed_dict=feed)
        train_total, _ = sess.run([loss, opt_total], feed_dict=feed)
        
        list_survey.append(train_survey)   #增添列表中的元素
        list_mobile.append(train_mobile)
        list_float.append(train_float)
        list_sensor.append(train_sensor)
        list_total.append(train_total)
        t2 = datetime.datetime.now()
        print('step =', nb_iter, '\n')
        print('train_survey =', train_survey, '\n')      
        print('train_mobile =', train_mobile, '\n')   
        print('train_float =', train_float, '\n')  
        print('train_sensor =', train_sensor, '\n')
        print('train_total =', train_total, '\n\n')
        dataframe = pd.DataFrame({'loss':list_total, 'loss_survey':list_survey, 'loss_mobile':list_mobile, 'loss_float':list_float, 'loss_sensor':list_sensor})   
        dataframe.to_csv("output_loss.csv", index=False)
        result = sess.run(merged, feed_dict=feed) #### Run all computational graphs
        writer.add_summary(result, nb_iter) ####
        writer.add_run_metadata(run_metadata, 'step{}'.format(nb_iter)) ####
        

    # output 
    output_results(sess,data,feed,graph)
    
    # convergence curve
    t1 = datetime.datetime.now()
    plt.tight_layout()
    plt.plot(list_survey,'r-')
    plt.plot(list_mobile,'b-')
    plt.plot(list_float,'g-')
    plt.plot(list_sensor,'k')
    plt.plot(list_total,'y-')
    plt.savefig('convergence curv.svg',dpi=300,format='svg')
    plt.show() 
    
    print('Training using Time: ',t1-t0,'\n')

    
    # Marginal analysis
    #margin_price_df,margin_avg_price_df=margin_price(link_set, path_set,od_set,feed,bottleneck)
    #margin_od_df=margin_od(link_set,path_set,od_set,feed)
    #margin_ozone_df=margin_ozone(link_set,path_set,od_set,ozone_set,feed)
    #congestion_component(link_set,path_set,od_set,ozone_set,feed,bottleneck)