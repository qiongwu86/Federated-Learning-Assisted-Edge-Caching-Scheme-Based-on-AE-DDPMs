import numpy as np
import pandas as pd
import torch
from data_set import DataSet
from user_info import UserInfo
import utils
from collections import Counter




def sampling_mobility(args,vehicle_num):
    """
    :param args
    :return: sample: matrix user_id|movie_id|rating|gender|age|occupation|label
    :return: user_group_train, the idx of sample for each client for training
    :return: user_group_test, the idx of sample for each client for testing
    """
    model_manager = utils.ModelManager('clients')
    '''Do you want to clean workspace and retrain model/clients again?'''
    '''if you want to change test_size or retrain model/clients, please set clean_workspace True'''
    model_manager.clean_workspace(args.clean_clients)
    try:
        users_group_train = model_manager.load_model(args.dataset + '-user_group_train')
        users_group_test = model_manager.load_model(args.dataset + '-user_group_test')
        sample = model_manager.load_model(args.dataset + '-sample')
    except OSError:
        ratings, user_info = get_dataset(args)
        users_num_client = np.random.randint(10,15,vehicle_num) 
        a=0
        for i in range(vehicle_num):
            a+=users_num_client[i]
        for i in range(vehicle_num):
            users_num_client[i] = int((user_info.index[-1] + 1) * users_num_client[i] / a )
        user_seq_client=[]
        for i in range(vehicle_num):
            num=0
            for j in range(i):
                num+=users_num_client[j]
            user_seq_client.append(num)
        # sample user_id|movie_id|rating|gender|age|occupation
        sample = pd.merge(ratings, user_info, on=['user_id'], how='inner')
        sample = sample.astype({'user_id': 'int64', 'movie_id': 'int64', 'rating': 'float64',
                                'gender': 'float64', 'age': 'float64', 'occupation': 'float64'})
        users_group_all, users_group_train, users_group_test ,request_content= {}, {}, {}, {}
        all_test_num = 0
        for i in range(vehicle_num):
            index_begin = ratings[ratings['user_id'] == user_seq_client[i] + 1].index[0]
            index_end = ratings[ratings['user_id'] == user_seq_client[i] + users_num_client[i] ].index[-1]
            users_group_all[i] = set(np.arange(index_begin, index_end + 1))
            NUM_train = int(0.7*len(users_group_all[i]))
            users_group_train[i] = set(np.random.choice(list(users_group_all[i]), NUM_train, replace=False))
            users_group_test[i] = users_group_all[i] - users_group_train[i]
            users_group_train[i] = list(users_group_train[i])
            users_group_test[i] = list(users_group_test[i])
            users_group_train[i].sort()
            users_group_test[i].sort()
            all_test_num += NUM_train/0.7*0.3
        # vehicle_request_num = dict([(k, []) for k in range(args.epochs)])
        #
        # for i in range(args.epochs):
        #     for j in range(vehicle_num):
        #         if j==0:
        #             vehicle_request_num[i]=[]
        #             request_content[i]=np.random.choice(list(users_group_test[j]), int(all_test_num / args.epochs * users_num_client[j] / a), replace=True)
        #             vehicle_request_num[i].append(int(all_test_num / 10 *users_num_client[j]/a))
        #
        #         else:
        #             request_content[i]=np.append(request_content[i],np.random.choice(list(users_group_test[j]), int(all_test_num / args.epochs *users_num_client[j]/a), replace=True))
        #             vehicle_request_num[i].append(int(all_test_num / 10 * users_num_client[j] / a))
        #
        #     request_content[i]=list(set(request_content[i]))
        #     request_content[i].sort()
        vehicle_request_num = np.random.randint(600,900,vehicle_num)

        model_manager.save_model(users_group_train, args.dataset + '-user_group_train')
        model_manager.save_model(users_group_test, args.dataset + '-user_group_test')
    sample = np.array(sample)
    print('Load Dataset Success')
    return sample, users_group_train, users_group_test, request_content, vehicle_request_num

def get_dataset(args):
    """
    :param: args:
    :return: ratings: dataFrame ['user_id' 'movie_id' 'rating']
    :return: user_info:  dataFrame ['user_id' 'gender' 'age' 'occupation']
    """
    model_manager = utils.ModelManager('data_set')
    user_manager = utils.UserInfoManager(args.dataset)

    '''Do you want to clean workspace and retrain model/data_set user again?'''
    '''if you want to retrain model/data_set user, please set clean_workspace True'''
    model_manager.clean_workspace(args.clean_dataset)
    user_manager.clean_workspace(args.clean_user)

    try:
        ratings = model_manager.load_model(args.dataset + '-ratings')
        print("Load " + args.dataset + " data_set success.\n")
    except OSError:
        ratings = DataSet.LoadDataSet(name=args.dataset)
        model_manager.save_model(ratings, args.dataset + '-ratings')
    try:
        user_info = user_manager.load_user_info('user_info')
        print("Load " + args.dataset + " user_info success.\n")
    except OSError:
        user_info = UserInfo.load_user_info(name=args.dataset)
        user_manager.save_user_info(user_info, 'user_info')

    return ratings, user_info

def cache_efficiency3(generated_ratings,test_idx,sample,k):

    top_k_values, top_k_indices = torch.topk(generated_ratings, k)
    top_k_indices += 1
    top_movie_indices = top_k_indices.numpy()
    test_dataset = sample[test_idx]
    requset_items = test_dataset[:, 1]
    count = Counter(requset_items)
    CACHE_HIT_NUM = 0
    for item in top_movie_indices:
        CACHE_HIT_NUM = CACHE_HIT_NUM + count[item]
        # print('CACHE_HIT_NUM:',CACHE_HIT_NUM,  count[item])
    CACHE_HIT_RATIO = CACHE_HIT_NUM / len(requset_items) * 100

    return CACHE_HIT_RATIO

def cache_efficiency2(generated_ratings,test_idx,sample,k_list):
    CACHE_HIT_RATIO = []
    Oracle_hit_ratio = []
    for i in range(len(k_list)):

        k = k_list[i]
        top_k_values, top_k_indices = torch.topk(generated_ratings, k)
        top_k_indices += 1
        top_movie_indices = top_k_indices.numpy()
        test_dataset = sample[test_idx]
        requset_items = test_dataset[:, 1]
        count = Counter(requset_items)

        CACHE_HIT_NUM = 0
        for item in top_movie_indices:
            CACHE_HIT_NUM = CACHE_HIT_NUM + count[item]
            # print('CACHE_HIT_NUM:',CACHE_HIT_NUM,  count[item])
        CACHE_HIT_RATIO.append(CACHE_HIT_NUM / len(requset_items) * 100)

        Oracle_hit = sum(list(sorted(count.values()))[-k:])
        Oracle_hit_ratio.append(Oracle_hit / len(requset_items) * 100)

    return CACHE_HIT_RATIO, Oracle_hit_ratio

def cache_efficiency(generated_ratings,test_idx,sample):
    k_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    CACHE_HIT_RATIO_all = []
    CACHE_HIT_RATIO_platoon_all = []
    CACHE_HIT_RATIO_rsu_all = []
    for i in range(len(k_list)):
        k = k_list[i]

        top_k_values_100, top_k_indices_100 = torch.topk(generated_ratings, 100)
        top_k_indices_100 += 1
        top_movie_indices_100 = top_k_indices_100.numpy()


        top_k_values, top_k_indices = torch.topk(generated_ratings, k)
        top_k_indices += 1
        top_movie_indices = top_k_indices.numpy()
        test_dataset = sample[test_idx]
        requset_items = test_dataset[:, 1]
        # print('requset_items',requset_items.shape,len(requset_items),type(requset_items),'\n',requset_items)

        count = Counter(requset_items)

        CACHE_HIT_NUM_platoon = 0
        for item in top_movie_indices_100:
            CACHE_HIT_NUM_platoon = CACHE_HIT_NUM_platoon + count[item]
            # print('CACHE_HIT_NUM:',CACHE_HIT_NUM,  count[item])
        CACHE_HIT_RATIO_platoon = CACHE_HIT_NUM_platoon / len(requset_items) * 100

        CACHE_HIT_NUM = 0
        for item in top_movie_indices:
            CACHE_HIT_NUM = CACHE_HIT_NUM + count[item]
        CACHE_HIT_RATIO = CACHE_HIT_NUM / len(requset_items) * 100
        CACHE_HIT_NUM_rsu = 0
        for item in top_movie_indices:
            if item not in top_movie_indices_100:
                CACHE_HIT_NUM_rsu = CACHE_HIT_NUM_rsu + count[item]
            # print('CACHE_HIT_NUM:',CACHE_HIT_NUM,  count[item])
        CACHE_HIT_RATIO_rsu = CACHE_HIT_NUM_rsu / len(requset_items) * 100
        Oracle_hit = sum(list(sorted(count.values()))[-k:])
        Oracle_hit_ratio = Oracle_hit / len(requset_items) * 100
        CACHE_HIT_RATIO_platoon_all.append(CACHE_HIT_RATIO_platoon)
        CACHE_HIT_RATIO_all.append(CACHE_HIT_RATIO)
        CACHE_HIT_RATIO_rsu_all.append(CACHE_HIT_RATIO_rsu)
    return CACHE_HIT_RATIO_platoon_all , CACHE_HIT_RATIO_rsu_all, CACHE_HIT_RATIO_all

def request_delay2(cache_hit_ratios, request_num, v2i_rate):

    # v2i_rate = np.mean(v2i_rate)
    comm_rate =  v2i_rate
    v2i_rate_mbs = 0.4 * v2i_rate
    request_delay_all = []
    for j in range(len(cache_hit_ratios)):
        cache_hit_ratio = cache_hit_ratios[j] / 100
        request_delay = 0
        request_delay += cache_hit_ratio * (request_num / comm_rate) * 800000
        request_delay += (1 - cache_hit_ratio) * (request_num / v2i_rate_mbs) * 800000
        request_delay_all.append(request_delay)
    return request_delay_all

def idx_train3(numbers):
    ''' 返回所给列表最大两个值的索引，以及其所占总体数据量的百分比 '''

    # 聚合权重计算
    agg_w = []
    total = sum(numbers)
    for i in range(len(numbers)):
        agg_w.append(numbers[i] / total)
    return agg_w

