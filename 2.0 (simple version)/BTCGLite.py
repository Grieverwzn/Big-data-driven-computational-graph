# In[0] Setting
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None # Do not show the copy warning
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
plt.rcParams['savefig.dpi']=100
plt.rcParams['figure.dpi']=100
import numpy as np
from keras import backend as K
NUM_PARALLEL_EXEC_UNITS = 8

config = tf.ConfigProto(intra_op_parallelism_threads = NUM_PARALLEL_EXEC_UNITS, 
         inter_op_parallelism_threads = 1, 
         allow_soft_placement = True, 
         device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })

session = tf.Session(config=config)

K.set_session(session)

import os

os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0" 


# In[0] Functions 
def connection(input_layer,trans_mat,inc_mat):
    sess=tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    tm=tf.multiply(trans_mat,inc_mat)
    trans_mat=tf.transpose(tf.transpose(tm)/tf.reduce_sum(tm,1))
    layer=tf.matmul(input_layer,trans_mat)
    return (tf.nn.relu(layer))

def init_variable(shape, initial_value, layer_name):
    estimation=tf.Variable(tf.random_normal(shape, mean=initial_value,stddev=0,dtype=tf.float64,seed=1),name=layer_name)
    return estimation


def build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile, f_count, lr_count,loss, learning_rate):
#def build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile,f_count, lr_count):
    t0 = datetime.datetime.now()
    my_opt_survey=tf.train.GradientDescentOptimizer(lr_survey)
    opt_survey = my_opt_survey.minimize(f_survey)
    t1 = datetime.datetime.now()
    print('Building Optimizer survey using Time: ',t1-t0,'\n')
    
    my_opt_mobile=tf.train.GradientDescentOptimizer(lr_mobile)
    opt_mobile = my_opt_mobile.minimize(f_mobile)
    t2 = datetime.datetime.now()
    print('Building Optimizer mobile using Time: ',t2-t0,'\n')
        
    my_opt_count=tf.train.GradientDescentOptimizer(lr_count)
    opt_count=my_opt_count.minimize(f_count)
    t3 = datetime.datetime.now()
    print('Building Optimizer sensor using Time: ',t3-t0,'\n')
     
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    opt_total=my_opt.minimize(loss)
    
    return opt_survey, opt_mobile, opt_count,opt_total


if __name__ == '__main__':
    # In[1] Input all the samples
    print ('----Step 1: Input data----','\n')
    t0 = datetime.datetime.now()
    all_node_df = pd.read_csv('input_node.csv',encoding='gbk')
    all_ozone_df = pd.read_csv('input_ozone.csv',encoding='gbk')
    all_od_df = pd.read_csv('input_od.csv',encoding='gbk')
    all_path_df = pd.read_csv('input_path.csv',encoding='gbk')
    all_link_df = pd.read_csv('input_link.csv',encoding='gbk')



    # In[2] Only consider the first sample
    ozone_df=all_ozone_df[all_ozone_df['sample_id']==1]
    od_df=all_od_df[all_od_df['sample_id']==1]
    path_df=all_path_df[all_path_df['sample_id']==1]
    link_df=all_link_df[all_link_df['sample_id']==1]

    link_with_sensor_df=link_df[link_df['sensor_count']>0]
    # It is important to reset the index
    ozone_df.reset_index(drop=True, inplace=True)
    od_df.reset_index(drop=True, inplace=True)
    path_df.reset_index(drop=True, inplace=True)
    link_df.reset_index(drop=True, inplace=True)




    # In[3] Basic parameters
    num_survey=all_ozone_df['sample_id'].max()
    num_mobile=all_od_df['sample_id'].max()
    num_float=all_path_df['sample_id'].max()
    num_sensor=all_link_df['sample_id'].max()
    num_ozone=ozone_df['ozone_id'].max()
    num_od=od_df['od_id'].max()
    num_path=path_df['path_id'].max()
    num_link=link_df['link_id'].max()


    batch_size=1

    t1 = datetime.datetime.now()
    print ('Input data using time', t1-t0,'\n')




    # In[4] Build up dicts

    print ('----Step 2: Build up Hash tables----','\n')
    t0 = datetime.datetime.now()
    print ('Dictionary on link layer...')
    link_df['link_pair'] = link_df.apply(lambda x: (int(x.from_node_id), int(x.to_node_id)), axis=1) # name each link
    link_id_pair_dict=link_df[['link_id','link_pair']].set_index('link_pair').to_dict()['link_id']

    print('Dictionary on ozone layer...')
    node_zone_dict=ozone_df[['node_id','ozone_id']].set_index('node_id').to_dict()['ozone_id']

    print('Dictionary on od layer...')
    od_df['od_pair']=od_df.apply(lambda x: (int(x.from_zone_id),int(x.to_zone_id)),axis=1)
    od_pair_dict=od_df[['od_pair','od_id']].set_index('od_pair').to_dict()['od_id']
    od_df['ozone_id']=od_df.apply(lambda x: node_zone_dict[int(x.from_zone_id)],axis=1)

    print('Dictionary from path to od...')
    path_df['od_id']=path_df.apply(lambda x: od_pair_dict[int(x.from_zone_id),int(x.to_zone_id)],axis=1)
    path_od_dict=path_df[['path_id','od_id']].set_index('path_id').to_dict()['od_id']

    print('Dictonary from od to o...')
    od_ozone_dict=od_df[['ozone_id','od_id']].set_index('od_id').to_dict()['ozone_id']

    t1 = datetime.datetime.now()
    print ('\n','CPU time:', t1-t0,'\n')




    # In[5] Build up Data set
    print ('----Step 3: Build up data sets----','\n')
    print ('Introduce samples...')
    t0 = datetime.datetime.now()
    survey_val=np.zeros(shape=[num_survey,num_ozone])
    for s in range(num_survey):
        survey_val[s]=all_ozone_df[all_ozone_df.sample_id==s+1].trip_generation

    sensor_val=np.zeros(shape=[num_sensor,num_link])
    for s in range(num_sensor):
        sensor_val[s]=all_link_df[all_link_df.sample_id==s+1].sensor_count

    mobile_val=np.zeros(shape=[num_mobile,num_ozone,num_od])
    for m in range(num_mobile):
        temp_matrix=np.zeros([num_ozone,num_od])
        for i in range(num_od):
            od_w=od_df.loc[i]
            temp_matrix[od_w.ozone_id-1][od_w.od_id-1]=od_w.OD_split
        mobile_val[m]=temp_matrix

    print ('Build up Placeholders...')
    survey_train=tf.placeholder(shape=[batch_size,num_ozone],dtype=tf.float64)
    mobile_train=tf.placeholder(shape=[batch_size,num_ozone,num_od],dtype=tf.float64)
    sensor_train=tf.placeholder(shape=[batch_size,num_link],dtype=tf.float64)    
    t1 = datetime.datetime.now()
    print ('\n','CPU time:', t1-t0,'\n')  
    


    # In[6] Build up incident matrices
    print ('Step 4: Build up incident matrices','\n')
    t0 = datetime.datetime.now()

    print('Link_id to path_id incident matrix...')
    path_link_inc_mat=np.zeros([num_path,num_link])
    for i in range(num_path):
            path_r = path_df.loc[i]   #循环取出path_df中的每一行
            node_list = path_r.node_sequence.split(';')     #取出除了倒数第一个元素外其他所有元素，用；隔开
            for link_l in range(len(node_list)-1):
                link_pair=(int(node_list[len(node_list)-1-link_l]),int(node_list[len(node_list)-2-link_l]))
                #link_pair=(int(node_list[link_l]),int(node_list[link_l+1]))
                link_id=link_id_pair_dict[link_pair]
                path_link_inc_mat[path_r.path_id-1][link_id-1]=1.0

    print('Path to od incident matrix...')
    od_path_inc_mat=np.zeros([num_od,num_path])       
    for i in range(num_path):
        path_r=path_df.loc[i]
        path_id=path_r.path_id
        od_id=path_od_dict[path_id]
        od_path_inc_mat[od_id-1][path_id-1]=1.0

    print('Od to ozone incident matrix...')
    ozone_od_inc_mat=np.zeros([num_ozone,num_od])       
    for i in range(num_od):
        od_w=od_df.loc[i]
        od_id=od_w.od_id
        ozone_id=od_ozone_dict[od_id]
        ozone_od_inc_mat[ozone_id-1][od_id-1]=1.0

    sensor_mat=np.array(link_df.sensor_count>=0).astype(float)
    sensor_mat=sensor_mat/np.sum(sensor_mat)
    t1 = datetime.datetime.now()
    print ('\n','CPU time:', t1-t0,'\n')       


    # In[8] Build up Computational graph
    print ('----Step 4: Build up Computational graph----','\n')
    print('Create layer from ozone to od...')
    est_alpha=init_variable([1,num_ozone], 300, 'alpha_')
    est_alpha=tf.nn.relu(est_alpha)
    est_gamma=init_variable([num_ozone,num_od], 1, 'gamma_')
    est_gamma=tf.nn.relu(est_gamma)
    est_q=connection(est_alpha,est_gamma,ozone_od_inc_mat)
    print('Create layer from od to path...')
    est_rou=init_variable([num_od,num_path], 0.5, 'rou_')
    est_rou=tf.nn.relu(est_rou)
    est_f=connection(est_q,est_rou,od_path_inc_mat)
    print('Create layer from path to link...')
    est_v=tf.nn.relu(tf.matmul(est_f,path_link_inc_mat))

    # In[9] Build up loss functions
    print ('----Step 5: Build up loss function----','\n')
    t0 = datetime.datetime.now()
    f_survey=tf.reduce_mean(tf.pow(tf.subtract(est_alpha,survey_train),2))
    #f_survey=tf.reduce_mean(tf.pow(tf.subtract(tf.divide(est_alpha,survey_train),1),2))
    #f_survey=tf.nn.softmax_cross_entropy_with_logits(labels=est_alpha, logits=survey_train)
    #f_mobile=tf.reduce_mean(tf.pow(tf.subtract(tf.divide(tf.multiply(est_gamma,ozone_od_inc_mat),mobile_train),1),2)) 
    f_mobile=tf.reduce_mean(tf.pow(tf.subtract(tf.multiply(est_gamma,ozone_od_inc_mat),mobile_train),2)) 
    #f_mobile=tf.nn.softmax_cross_entropy_with_logits(labels=tf.multiply(est_gamma,ozone_od_inc_mat), logits=mobile_train)
    #f_count=tf.reduce_mean(tf.pow(tf.subtract(tf.divide(tf.multiply(est_v,sensor_mat),tf.multiply(sensor_train,sensor_mat)),1),2))
    f_count=tf.reduce_mean(tf.pow(tf.subtract(tf.multiply(est_v,sensor_mat),tf.multiply(sensor_train,sensor_mat)),2))
        #f_count=tf.nn.softmax_cross_entropy_with_logits(labels=tf.multiply(est_v,sensor_mat), logits=tf.multiply(sensor_train,sensor_mat))
    #loss = tf.reduce_sum(tf.stack([f_survey, f_mobile,f_count]))
    loss = f_survey + f_mobile + f_count


    maximum_iterations=600
    curr_iter = tf.Variable(0) 
    lr_survey = tf.train.exponential_decay(0.5, curr_iter, decay_steps=maximum_iterations, decay_rate=0.99)
    lr_mobile = tf.train.exponential_decay(0.1, curr_iter, decay_steps=maximum_iterations/5.3, decay_rate=0.99)
    lr_count = tf.train.exponential_decay(0.01, curr_iter, decay_steps=maximum_iterations, decay_rate=0.99)
    learning_rate = tf.train.exponential_decay(0.01, curr_iter, decay_steps=maximum_iterations, decay_rate=0.99)
    opt_survey, opt_mobile, opt_count,opt_total = build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile, f_count, lr_count,loss,learning_rate)
    #opt_survey, opt_mobile, opt_count,opt_total = build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile, f_count, lr_count,loss,learning_rate)

    t1 = datetime.datetime.now()
    print ('\n','CPU time:', t1-t0,'\n') 
    # In[8] Build up loss function
    #config = tf.ConfigProto(intra_op_parallelism_threads=0, 
    #                        inter_op_parallelism_threads=0, 
    #                        allow_soft_placement=True)



    print ('----Step 6: Start training----','\n')
    tt0 = datetime.datetime.now()
    with tf.Session(config=config) as sess:
        
        init=tf.global_variables_initializer()
        sess.run(init)
        list_survey=[]
        list_mobile=[]
        list_sensor=[]
        list_total=[]
        for nb_iter in range(maximum_iterations):
        # (1) Select a sample randomly
            t0 = datetime.datetime.now()
            rand_survey_index=np.random.choice(num_survey,batch_size)
            rand_mobile_index=np.random.choice(num_mobile,batch_size)
            rand_sensor_index=np.random.choice(num_sensor,batch_size)    
            feed={}
            feed[survey_train]=survey_val[rand_survey_index]
            feed[mobile_train]=mobile_val[rand_mobile_index]
            feed[sensor_train]=sensor_val[rand_sensor_index]
            
            train_total, _ = sess.run([loss, opt_total], feed_dict=feed)    
            train_survey, _ = sess.run([f_survey, opt_survey], feed_dict=feed) ####
            train_mobile, _ = sess.run([f_mobile, opt_mobile], feed_dict=feed)
            train_sensor, _ = sess.run([f_count, opt_count], feed_dict=feed)
            #train_total, _ = sess.run([loss, opt_total], feed_dict=feed)    
            train_total=train_survey+train_mobile+train_sensor

            list_survey.append(train_survey)   #增添列表中的元素
            list_mobile.append(train_mobile)
            list_sensor.append(train_sensor)
            list_total.append(train_total)
            
            print('step',nb_iter,':survey error=', train_survey)
            print('step',nb_iter,':mobile error=', train_mobile)
            print('step',nb_iter,':sensor error=', train_sensor)
            print('step',nb_iter,':total error=', train_total)
            #print(np.round(sess.run(est_rou),2))
            output_link_flow=sess.run(est_v)
            output_path_flow=sess.run(est_f)
            output_path_proportion=sess.run(est_rou)
            output_od_flow=sess.run(est_q)
            output_od_distribution=sess.run(est_gamma)
            output_ozone_generation=sess.run(est_alpha)
            t1 = datetime.datetime.now()
            print ('\n','CPU time:', t1-t0,'\n')

    tt1 = datetime.datetime.now()         
    print ('\n','CPU time:', tt1-tt0,'\n')

    # font2 = {'family' : 'Arial',
    # 'weight' : 'normal',
    # 'size'   : 30,
    # }
    #plt.xlabel('iteration numbers',font2)
    #plt.ylabel('loss erros',font2)  
    #plt.figure(figsize=(20,10))
    plt.plot(list_survey,'r-')
    plt.savefig('convergence_survey.png',dpi=300,format='png')
    plt.show()
    plt.plot(list_mobile,'b-')
    plt.savefig('convergence_mobile.png',dpi=300,format='png')
    plt.show()
    plt.plot(list_sensor,'k')
    plt.savefig('convergence_sensor.png',dpi=300,format='png')
    plt.show()
    plt.plot(list_total,'y-')
    plt.savefig('convergence.png',dpi=300,format='png')
    plt.show()

