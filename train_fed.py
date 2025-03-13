import copy
import torch
import time
from torch.utils.data import DataLoader
from model_ae import train_autoencoder, train_ddpm, aggregate_model_weight, aggregate_model_weights


def train_hfl(args,platoon_train_idx,train_data,gl_dm, gl_ae):
    client_epoch_time=[]
    ae_weights = []
    dm_weights = []
    for i in range(args.clients_num):
        start_time = time.time()
        ae_dataloader = DataLoader(train_data[i][:, 1:-3], batch_size=64, shuffle=True)
        ae_model = copy.deepcopy(gl_ae)
        train_ae, ae_avg_loss = train_autoencoder(ae_model, args, ae_dataloader, device=torch.device('cpu'))
        agg_ae_w = train_ae.state_dict()
        train_ae.eval()
        with torch.no_grad():
            dim_data = train_ae.encode(train_data[i][:, 1:-3])
        tra_dm_load = DataLoader(dim_data, batch_size=args.batch_size, shuffle=True)
        dm_model = copy.deepcopy(gl_dm)
        dm_avg_loss, train_dm = train_ddpm(args,tra_dm_load, dm_model)
        train_dm_w = train_dm.state_dict()
        agg_ae_w = [agg_ae_w, platoon_train_idx[i]]
        train_dm_w = [train_dm_w, platoon_train_idx[i]]
        ae_weights.append(agg_ae_w)
        dm_weights.append(train_dm_w)
        client_epoch_time.append(time.time() - start_time)
        print(f"- User {i+1} - AutoEncoder Training - Local Epoch {args.AE_local_ep}- Loss: {ae_avg_loss:.4f}")
        print(f"- User {i+1} ---- DDPM Training ----- Local Epoch {args.dm_lr_ep}- Loss: {dm_avg_loss:.4f}")
    agg_ae_w = aggregate_model_weight(ae_weights)
    agg_dm_w = aggregate_model_weight(dm_weights)
    return agg_ae_w,agg_dm_w, max(client_epoch_time)

def train_hie_fed(dict_AE_model, platoon_model_dict,platoon_idx, args,platoon_train_idx,train_data,gl_dm, gl_ae):

    client_epoch_time=[]
    ae_weights = []
    dm_weights = []
    for i in range(len(platoon_model_dict)):
        start_time = time.time()
        ae_dataloader = DataLoader(train_data[i][:, 1:-3], batch_size=64, shuffle=True)
        ae_model = copy.deepcopy(dict_AE_model[i][-1])
        train_ae, ae_avg_loss = train_autoencoder(ae_model, args, ae_dataloader, device=torch.device('cpu'))
        dict_AE_model[i].append(copy.deepcopy(train_ae))
        agg_ae_w = train_ae.state_dict()
        train_ae.eval()
        with torch.no_grad():
            dim_data = train_ae.encode(train_data[i][:, 1:-3])
        tra_dm_load = DataLoader(dim_data, batch_size=args.batch_size, shuffle=True)
        dm_model = copy.deepcopy(platoon_model_dict[i][-1])
        dm_avg_loss, train_dm = train_ddpm(args,tra_dm_load, dm_model)
        platoon_model_dict[i].append(copy.deepcopy(train_dm))
        train_dm_w = train_dm.state_dict()


        ae_weights.append(agg_ae_w)
        dm_weights.append(train_dm_w)
        client_epoch_time.append(time.time() - start_time)

        print(f" - Platoon {platoon_idx+1} - Vehicle {i+1} - AutoEncoder Training - Local Epoch {args.AE_local_ep}- Loss: {ae_avg_loss:.4f}")
        print(f" - Platoon {platoon_idx+1} - Vehicle {i+1} ---- DDPM Training ----- Local Epoch {args.dm_lr_ep}- Loss: {dm_avg_loss:.4f}")
    agg_ae_w = aggregate_model_weights(gl_ae, ae_weights, platoon_train_idx[1])
    gl_ae.load_state_dict(agg_ae_w)
    agg_dm_w = aggregate_model_weights(gl_dm, dm_weights, platoon_train_idx[1])
    gl_dm.load_state_dict(agg_dm_w)
    for i in range(len(dict_AE_model)):
        dict_AE_model[i].append(copy.deepcopy(gl_ae))
    for i in range(len(platoon_model_dict)):
        platoon_model_dict[i].append(copy.deepcopy(gl_dm))

    return gl_ae,gl_dm,dict_AE_model,platoon_model_dict,max(client_epoch_time)

