import numpy as np
import torch
from itertools import chain
from options import args_parser
import matplotlib.pyplot as plt
from dataset_processing import cache_efficiency, sampling_mobility,cache_efficiency3,idx_train3,request_delay2
from data_set import convert
from model_ddpm import MLPDiffusion, GaussianMultinomialDiffusion
from model_ae import generator_data, AutoEncoder
from train_fed import train_hfl
from env_communicate import Environ


if __name__ == '__main__':
    args = args_parser()
    in_out_dim = args.in_out_dim
    client_num = args.clients_num
    hidden_dim = 256
    num_step = args.num_step

    idx=0
    # gpu or cpu
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    # load sample users_group_train users_group_test
    sample, users_group_train, users_group_test, request_content, user_request_num = sampling_mobility(args, args.clients_num)
    test_idx = []
    for i in range(client_num):
        test_idx.append(users_group_test[i])
    test_idx = list(chain.from_iterable(test_idx))

    time_slow = 0.1
    user_dis = np.random.randint(0,500,client_num)
    env = Environ(client_num)
    env.new_random_game(user_dis)  # initialize parameters in env

    model = MLPDiffusion(in_out_dim, hidden_dim, in_out_dim, num_step)
    diffusion_model = GaussianMultinomialDiffusion(
        num_numerical_features=in_out_dim,
        denoise_fn=model,
        num_timesteps=num_step,
        device=torch.device('cpu')
    )
    gl_dm = diffusion_model
    train_idx1 = idx_train3(user_request_num)
    gl_ae = AutoEncoder(input_dim=3952, hidden_dim=100, latent_dim=in_out_dim)
    Train_data1 = []
    for i in range(client_num):
        num = np.random.randint(8000,10000)
        train_idx = users_group_train[i][:num]
        train_data = convert(sample[train_idx], int(max(sample[:, 1])))
        train_data = torch.Tensor(train_data).float()
        Train_data1.append(train_data)

    cache_hit_ratio_500 = []
    cache_hit_ratio_100 = []
    client_epoch_time_all = []
    ae_am_delay = []
    ae_am_delay_100 = []

    v2i_rate = env.Compute_Performance_Train_mobility(client_num)
    while idx < args.epochs:
        print(f' | Global Training Round : {idx + 1}')
        gl_ae_w1,gl_dm_w1,client_epoch_time1 = train_hfl(args,train_idx1,Train_data1,gl_dm, gl_ae)
        client_epoch_time_all.append(client_epoch_time1)
        gl_ae.load_state_dict(gl_ae_w1)
        gl_dm.load_state_dict(gl_dm_w1)
        idx += 1
        generated_data = generator_data(gl_dm, 1000)
        gl_ae.eval()
        with torch.no_grad():
            generated_data_original_dim = gl_ae.decoder(generated_data)
        generated_ratings = torch.sum(generated_data_original_dim, dim=0)
        cache_hit_ratio_500.append(cache_efficiency3(generated_ratings,test_idx,sample,500))
        cache_hit_ratio_100.append(cache_efficiency3(generated_ratings,test_idx,sample,100))
        print('-------------------------------------------------------------------------------------------------------')

        if idx == args.epochs:
            generated_data = generator_data(gl_dm, 1000)
            gl_ae.eval()
            with torch.no_grad():
                generated_data_original_dim = gl_ae.decoder(generated_data)

            generated_ratings = torch.sum(generated_data_original_dim, dim=0)

            CACHE_HIT_RATIO_platoon , CACHE_HIT_RATIO_rsu, CACHE_HIT_RATIO_all = cache_efficiency(generated_ratings, test_idx, sample)


            Oracle_hit_ratio_avg = [10.957230142566193, 18.350973256318653, 24.5133718406731, 29.886815131381255,
                                    34.704350439050444, 39.028079196020165, 42.961169910854395, 46.592434309371974,
                                    49.970618677172716, 53.08637441153885]
            AE_DDPM_cache_avg = [10.870755567426798, 18.108911221662048, 24.2382558178358, 29.472471703782844,
                                 34.24293011919469, 38.566992754832896, 42.483055657574035, 46.203799539247434,
                                 49.392006944676304, 52.40626356382091]
            GAN_cache_efficiency_avg = [10.54088344295683, 17.628459817702247, 23.200227037494574, 28.555307001435676,
                                        33.134119061133184, 37.32963840940202, 41.11682414610531, 44.58649126907282,
                                        48.14897666188107, 51.16356715969417]
            thompson_sampling_avg = [2.2883472812204517, 5.796723069758252, 9.761639311872875, 15.27011388814406,
                                     19.237499823593332, 21.998969785066116, 27.92270565489211, 31.05251273656134,
                                     34.182672631564095, 39.360913927659155]
            Greedy_cache_efficiency_avg = [10.068963172596709, 16.78049929370213, 22.460309786555985, 27.45374808334695,
                                           31.89512928884481, 35.936005319081715, 39.52388723457956, 42.97194680504206,
                                           46.04144937882045, 48.935819930405444]

            request_number = sum(user_request_num)/len(user_request_num)
            v2i_rate_avg = sum(v2i_rate)/len(v2i_rate)
            ae_am_delay_100 = request_delay2(cache_hit_ratio_100, request_number, v2i_rate_avg)
            Oracle_request_delay = request_delay2(Oracle_hit_ratio_avg,request_number, v2i_rate_avg)
            ae_dm_request_delay = request_delay2(AE_DDPM_cache_avg,request_number, v2i_rate_avg)
            WGAN_request_delay = request_delay2(GAN_cache_efficiency_avg,request_number,v2i_rate_avg)
            Greedy_request_delay = request_delay2(Greedy_cache_efficiency_avg,request_number,v2i_rate_avg)
            TS_request_delay = request_delay2(thompson_sampling_avg,request_number,v2i_rate_avg)
            print('Oracle_request_delay',Oracle_request_delay)
            print('ae_dm_request_delay',ae_dm_request_delay)
            print('WGAN_request_delay',WGAN_request_delay)
            print('Greedy_request_delay',Greedy_request_delay)
            print('TS_request_delay',TS_request_delay)
            print('AE_DDPM_cache_efficiency:',CACHE_HIT_RATIO_all)
            print('\ncache_hit_ratio_500',cache_hit_ratio_500)
            print('\ncache_hit_ratio_100',cache_hit_ratio_100)
            print('\nae_am_delay_100',ae_am_delay_100)
        if idx > args.epochs:
            break

ax=plt.gca()
ax.yaxis.set_minor_locator(plt.MultipleLocator(2))
ax.grid(which="both",linestyle='--')
plt.xlim(50, 500)
plt.ylim(24, 38)
plt.xlabel('Cache capacity',fontsize=16)
plt.ylabel('Request content delay (ms)',fontsize=16)
cache_size=[50,100,150,200,250,300,350,400,450,500]
# Oracle_hit_ratio_avg
plt.plot(cache_size, Oracle_request_delay, color='#DE582B', marker='o', linewidth=1.5, linestyle='-', label='Oracle')

plt.plot(cache_size, ae_dm_request_delay, color='#1868B2', marker='^', linewidth=1.5, linestyle='-', label='Ours')

plt.plot(cache_size, WGAN_request_delay, color='#018A67', marker='x',linewidth=1.5, linestyle='-',label='CPPPP')
# N-ε-greedy
plt.plot(cache_size, Greedy_request_delay,   color='#F3A332', marker='v',linewidth=1.5, linestyle='-', label='N-ε-greedy')
# Thompson_sampling
plt.plot(cache_size, TS_request_delay, color='deeppink', linewidth=1.5, marker='*', linestyle='-', label='Thompson sampling')
plt.legend(prop={'family' : 'Times New Roman', 'size' : 12})
plt.savefig("2.pdf", format="pdf")  # bbox_inches="tight" 去除多余的白边
plt.show()

