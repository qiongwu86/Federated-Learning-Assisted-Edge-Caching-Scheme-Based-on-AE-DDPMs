import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 计算list均值
def list_avg(list1,list2,list3,list4,list5):
    list_all = []
    list_all.extend([list1,list2,list3,list4,list5])
    columns = zip(*list_all)
    column_sums = [sum(column) for column in columns]
    column_sums = [num / 5 for num in column_sums]

    return column_sums

'''
图一不同模型训练的缓存命中率的对比
'''

Oracle_hit_ratio_avg = [10.957230142566193, 18.350973256318653, 24.5133718406731, 29.886815131381255, 34.704350439050444, 39.028079196020165, 42.961169910854395, 46.592434309371974, 49.970618677172716, 53.08637441153885]
AE_DDPM_cache_avg = [10.870755567426798, 18.108911221662048, 24.2382558178358, 29.472471703782844, 34.24293011919469, 38.566992754832896, 42.483055657574035, 46.203799539247434, 49.392006944676304, 52.40626356382091]
GAN_cache_efficiency_avg = [10.54088344295683, 17.628459817702247, 23.200227037494574, 28.555307001435676, 33.134119061133184, 37.32963840940202, 41.11682414610531, 44.58649126907282, 48.14897666188107, 51.16356715969417]

thompson_sampling_avg = [2.2883472812204517, 5.796723069758252, 9.761639311872875, 15.27011388814406, 19.237499823593332, 21.998969785066116, 27.92270565489211, 31.05251273656134, 34.182672631564095, 39.360913927659155]
Greedy_cache_efficiency_avg = [10.068963172596709, 16.78049929370213, 22.460309786555985, 27.45374808334695, 31.89512928884481, 35.936005319081715, 39.52388723457956, 42.97194680504206, 46.04144937882045, 48.935819930405444]

# plt cache hit ratio
# plt.figure(figsize=(6.25, 5))
ax=plt.gca()
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
ax.grid(which="both",linestyle='--')
# 设置坐标轴范围、名称
plt.xlim(50, 500)
plt.ylim(0, 55)
plt.xlabel('Cache capacity',fontsize=16)
plt.ylabel('Cache hit percentage',fontsize=16)
cache_size=[50,100,150,200,250,300,350,400,450,500]

# Oracle_hit_ratio_avg
plt.plot(cache_size, Oracle_hit_ratio_avg, color='#DE582B', marker='o', linewidth=1.5, linestyle='-', label='Oracle')
# AE_DDPM_cache_avg
plt.plot(cache_size, AE_DDPM_cache_avg, color='#1868B2', marker='^', linewidth=1.5, linestyle='-', label='Ours')
# # GAN_cache_efficiency_avg
plt.plot(cache_size, GAN_cache_efficiency_avg, color='#018A67',   marker='x',linewidth=1.5, linestyle='-',label='CPPPP')
# N-ε-greedy
plt.plot(cache_size, Greedy_cache_efficiency_avg,   color='#F3A332',        marker='v',linewidth=1.5, linestyle='-', label='N-ε-greedy')
# Thompson_sampling
plt.plot(cache_size, thompson_sampling_avg, color='deeppink', linewidth=1.5, marker='*', linestyle='-', label='Thompson sampling')

plt.legend(prop={'family' : 'Times New Roman', 'size' : 12})
plt.savefig("1.pdf", format="pdf")  # bbox_inches="tight" 去除多余的白边
# plt.show()

'''
图二缓存命中率和内容传输时延随round变化
'''
cache_hit_100 = [5.779420455342208, 11.998557648967816, 15.268554410355545, 16.8511340151113, 17.01874054709176, 17.368644223417505, 17.572309531202528, 17.696845858749764, 17.887489941938696, 17.797676879981037, 17.88114626841752, 17.782652390062466, 17.853768309010356, 17.957938105779085, 17.974298105912638, 17.981643412095046, 17.99967279999733, 17.870796064251397, 17.989322595831204, 18.10317484165857, 17.916203411560844, 18.132556066388215, 18.022042596098306, 18.014697289915897, 17.984314432525018, 18.00434708574977, 17.964281779300258, 17.84575524772045, 18.07012096383772, 17.966285044622733]

ae_am_delay_100 = [56.31053617396357, 54.1338381561946, 52.9893392897089, 52.435436428044376, 52.376774141851214, 52.25430785513721, 52.18302499741245, 52.13943728277092, 52.072711853654795, 52.10414642533997, 52.07493213938721, 52.10940499681148, 52.08451442517971, 52.048054996310654, 52.04232899626392, 52.039758139100066, 52.03344785333427, 52.07855471084535, 52.03707042479242, 51.99722213875284, 52.06266213928705, 51.98693871009746, 52.025618424698926, 52.02818928186277, 52.03882328194958, 52.03181185332092, 52.04583471057825, 52.08731899663118, 52.00879099599014, 52.04513356771538]


fig,ax1=plt.subplots()
ax2=ax1.twinx()
plt.gcf().set_facecolor('white')
# ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax1.grid(ls='--')
ax1.plot(range(1,31),cache_hit_100, color='royalblue', linewidth=1.5, linestyle='-')
ax1.scatter(range(1,31), cache_hit_100, s=10, marker='o', color='royalblue')
ax1.set_xlabel('Training round',fontsize=18)
ax1.set_ylabel('Cache hit percentage',color='royalblue',fontsize=18)

"-------------------------------------------"
"坐标轴范围"
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 20)
"-------------------------------------------"
ax2.plot(range(1,31), ae_am_delay_100, color='tomato', linewidth=1.5, linestyle='-')
ax2.scatter(range(1,31), ae_am_delay_100, s=10, marker='*', color='tomato')
ax2.set_ylabel('request content delay(ms)', color='tomato',fontsize=18)
y1_major_locator = MultipleLocator(5)

"-------------------------------------------"
"坐标轴范围"
ax2.set_ylim(50,60)
"-------------------------------------------"
ax1.yaxis.set_major_locator(y1_major_locator)
plt.savefig("3.pdf", format="pdf")

plt.show()

'''
图三缓存命中率随参加用户数的变化
'''

cache_hit_number = [7.791922007428499, 11.036943184849294,14.937684628875122,17.115555170275336,17.49562425398265, 17.953001113622875 ]

plt.ylim(0,20)
plt.xlabel('The number of user',fontsize=18)
plt.ylabel('Cache hit percentage',fontsize=18)
name_list=['5', '10', '15', '20', '25', '30']
ax=plt.gca()
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
ax.grid(which="both",linestyle='--',zorder=0)
plt.bar(np.arange(len(cache_hit_number)),cache_hit_number, width=0.8,color='royalblue',zorder=10,tick_label=name_list)
plt.plot(np.arange(len(cache_hit_number)),cache_hit_number, color='red',marker='.',zorder=10)
plt.savefig("4.pdf", format="pdf")

plt.show()

'''
图四 训练round随step的变化
'''

cache_hit_100 = [0,5.779420455342208, 11.998557648967816, 15.268554410355545, 16.8511340151113, 17.01874054709176, 17.368644223417505, 17.572309531202528, 17.696845858749764, 17.887489941938696, 17.797676879981037, 17.88114626841752, 17.782652390062466, 17.853768309010356, 17.957938105779085, 17.974298105912638, 17.981643412095046, 17.99967279999733, 17.870796064251397, 17.989322595831204, 18.10317484165857, 17.916203411560844, 18.132556066388215, 18.022042596098306, 18.014697289915897, 17.984314432525018, 18.00434708574977, 17.964281779300258, 17.84575524772045, 18.07012096383772, 17.966285044622733]

cache_dm_100 = [0,3.2181577464225093, 2.4257906163188943, 2.4654756703939493, 2.9363605137046833, 2.6932478715138015, 2.1743407778937573, 2.640223303463938, 2.085299522111912, 2.3360823848383085, 2.0479488829824484, 2.112979013609639, 1.9975922177275471, 2.4017794911642394, 3.39590677013683, 2.794961665571715, 2.278722474746633, 2.7439380246180733, 2.3427521418257125, 2.893007093286556, 2.5915340774558877, 2.266383424319935, 2.4961565525360085, 2.5795285148785605, 2.9543688575706746, 2.61054288486999, 2.521835116937514, 2.045614468036857, 2.4421315209380348, 2.879667579311748, 2.711923191078533]


# plt.figure(figsize=(6.25, 5))
ax=plt.gca()
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
ax.grid(which="both",linestyle='--')
# 设置坐标轴范围、名称
plt.xlim(0, 30)
plt.ylim(0, 20)
plt.xlabel('Training round',fontsize=18)
plt.ylabel('Cache hit percentage',fontsize=18)

# AE_DDPM_cache_avg
plt.plot(range(31), cache_hit_100, color='#1868B2', marker='^', linewidth=1.5, linestyle='-', label='AE-DDPMs')

plt.plot(range(31), cache_dm_100, color='deeppink', linewidth=1.5, marker='*', linestyle='-', label='DDPMs')

plt.legend(prop={'family' : 'Times New Roman', 'size' : 18})
plt.savefig("5.pdf", format="pdf")
plt.show()
