import numpy as np
import os
import copy
import torch
from torch import nn
import time
import itertools

start_time_global = time.time()

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

def update_npz_file(npz_filename, new_arrays):
    if not os.path.exists(npz_filename):
        np.savez(npz_filename)
    with np.load(npz_filename, allow_pickle=True) as data:
        existing_arrays = {key: data[key] for key in data.files}
    updates_made = False
    for array_name, array_data in new_arrays.items():
        if array_name in existing_arrays:
            existing_arrays[array_name] = array_data
            updates_made = True
        else:
            existing_arrays[array_name] = array_data
            updates_made = True
    if updates_made:
        np.savez(npz_filename, **existing_arrays)

def get_latency(num_worker, worker_selection_binary, eta_W, channel_gain, comp_overhead, comm_overhead, P_U_Watt, P_W_Watt, B):
    t_Encode = np.zeros((num_worker))
    t_Comp = np.zeros((num_worker))
    t_Upload = np.zeros((num_worker))
    t_MAP_1 = comp_overhead['Map_1'] / f_C_user
    t_Mask = comp_overhead['Mask'] / f_C_user
    t_Decode = comp_overhead['Decode'] / f_C_user
    t_Map_2 = comp_overhead['Map_2'] / f_C_user
    t_final = comp_overhead['Final'] / f_C_user
    R_Multicast = B * np.log2(1 + np.ma.min(np.ma.masked_where(worker_selection_binary == 0, channel_gain)) * P_U_Watt / Noise_Watt)
    t_Multicast = comm_overhead['Multicast'] / R_Multicast
    for j in range(num_worker):
        t_Encode[j] = worker_selection_binary[j] * comp_overhead['Encode'] / f_C_worker[j]
        t_Comp[j] = worker_selection_binary[j] * comp_overhead['Comp'] / f_C_worker[j]
        if worker_selection_binary[j] == 1:
            R_Upload = eta_W[j] * B * np.log2(1 + channel_gain[j] * P_W_Watt[j] / Noise_Watt)
            t_Upload[j] = comm_overhead['Upload'] / R_Upload
    objective = t_MAP_1 + t_Mask + np.max(np.multiply(worker_selection_binary, t_Multicast + t_Encode + t_Comp + t_Upload)) + t_Decode + t_Map_2 + t_final
    system_latency = objective
    comp_latency = {
        'Map_1': t_MAP_1,
        'Mask': t_Mask,
        'Encode': t_Encode,
        'Comp': t_Comp,
        'Decode': t_Decode,
        'Map_2': t_Map_2,
        'Final': t_final,
    }
    comm_latency = {
        'Multicast': t_Multicast,
        'Upload': t_Upload,
    }
    return objective, system_latency, comp_latency, comm_latency

def num_cycle(field='Real', opera='Add', dtype='Float'):
    num_Cycle_Modulo = 40
    cycle_times = {
        'Real': {
            'Float': {'Add': 3, 'Sub': 3, 'Mul': 5, 'Div': 15, 'Cmp': 3, 'Round': 21},
            'Int': {'Add': 1, 'Sub': 1, 'Mul': 3, 'Div': 40, 'Cmp': 1}
        },
        'FF': {
            'Add': None,
            'Sub': None,
            'Mul': None
        }
    }

    if field == 'FF':
        int_cycles = cycle_times['Real']['Int']
        return int_cycles.get(opera, None) + num_Cycle_Modulo if opera in int_cycles else None

    return cycle_times.get(field, {}).get(dtype, {}).get(opera, None)

def get_overhead(l):

    Lambda_Comp_PiNet = l * (
        16 * 16 * 64 * 147 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        16 * 16 * 64 * 146 * num_cycle(field='Real', opera='Add', dtype='Float') +
        16 * 16 * 64 * 576 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        16 * 16 * 64 * 575 * num_cycle(field='Real', opera='Add', dtype='Float') +
        16 * 16 * 64 * 576 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        16 * 16 * 64 * 575 * num_cycle(field='Real', opera='Add', dtype='Float') +
        16 * 16 * 64 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        16 * 16 * 64 * 2 * num_cycle(field='Real', opera='Add', dtype='Float') +
        10 * 16384 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        (10 * (16384 - 1) + 10) * num_cycle(field='Real', opera='Add', dtype='Float')
    )
    Lambda_final = (10 - 1) * l * num_cycle(field='Real', opera='Cmp', dtype='Float')
    Lambda_Comp = Lambda_Comp_PiNet + Lambda_final
    
    return Lambda_Comp

def get_reward(objective):
    return np.exp(- objective/1e8 * 0.1e-2)

def train(model, experience):
    num_iter = 1000
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
    length = len(experience)
    index = np.arange(length)
    np.random.shuffle(index)
    cnt = 0
    tot_loss = 0
    while True:
        epoch_loss = 0
        for idx in index:
            state = torch.tensor(experience[idx][0], dtype=torch.float32)
            reward = torch.tensor(experience[idx][1], dtype=torch.float32)
            optimizer.zero_grad()
            delta = model(state) - reward
            loss = delta * delta
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            tot_loss += loss.item()
            cnt += 1
            if cnt >= num_iter:
                return copy.deepcopy(model), tot_loss / num_iter
        if epoch_loss / length <= 1e-3:
            return copy.deepcopy(model), epoch_loss / length

def compute_gradient_single_output(model, x):
    output = model(x)
    model.zero_grad()
    output.backward(retain_graph=True)
    return torch.cat([p.grad.flatten().detach() for p in model.parameters()]).cpu()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_storage(num_episode, num_epoch, num_worker):
    solution = {
        'K': np.zeros((num_episode, num_epoch), dtype=int),
        'a': np.zeros((num_episode, num_epoch, num_worker), dtype=int),
        'eta_W': np.zeros((num_episode, num_epoch, num_worker))
    }
    latency = {
        'Objective': np.zeros((num_episode, num_epoch)),
        'System_Latency': np.zeros((num_episode, num_epoch)),
        'Map_1': np.zeros((num_episode, num_epoch)),
        'Mask': np.zeros((num_episode, num_epoch)),
        'Encode': np.zeros((num_episode, num_epoch, num_worker)),
        'Comp': np.zeros((num_episode, num_epoch, num_worker)),
        'Decode': np.zeros((num_episode, num_epoch)),
        'Map_2': np.zeros((num_episode, num_epoch)),
        'Final': np.zeros((num_episode, num_epoch)),
        'Multicast': np.zeros((num_episode, num_epoch, num_worker)),
        'Upload': np.zeros((num_episode, num_epoch, num_worker))
    }
    return solution, latency

def update_solution(solution, K, worker_selection, bandwidth_allocation, episode):
    solution['K'][episode] = K
    solution['a'][episode] = worker_selection
    solution['eta_W'][episode] = bandwidth_allocation
    return solution

def update_latency(latency, objective, system_latency, comp_latency, comm_latency, episode):
    latency['Objective'][episode] = objective
    latency['System_Latency'][episode] = system_latency
    latency['Map_1'][episode] = comp_latency['Map_1']
    latency['Mask'][episode] = comp_latency['Mask']
    latency['Encode'][episode] = comp_latency['Encode']
    latency['Comp'][episode] = comp_latency['Comp']
    latency['Decode'][episode] = comp_latency['Decode']
    latency['Map_2'][episode] = comp_latency['Map_2']
    latency['Final'][episode] = comp_latency['Final']
    latency['Multicast'][episode] = comm_latency['Multicast']
    latency['Upload'][episode] = comm_latency['Upload']
    return latency

def update_name(dictionary, new_name_1, new_name_2=None):
    if new_name_2 is None:
        updated_dictionary = {}
        for key in dictionary:
            updated_key = key + new_name_1
            updated_dictionary[updated_key] = dictionary[key]
    else:
        updated_dictionary = {}
        for key in dictionary:
            if key == 'Objective' or key == 'System_Latency':
                updated_key = key + new_name_2
            else:
                updated_key = new_name_1 + key + new_name_2
            updated_dictionary[updated_key] = dictionary[key]
    return updated_dictionary

def generate_valid_combinations(M, K, r, T):
    a_values = list(itertools.product([0, 1], repeat=M))
    one_hot_vectors = np.eye(len(K))
    valid_combinations = []
    for k_index, k in enumerate(K):
        one_hot = one_hot_vectors[k_index]
        for a in a_values:
            if sum(a) == r * (k + T - 1) + 1:
                combined_vector = np.concatenate((one_hot, a))
                valid_combinations.append(combined_vector)
    valid_combinations_array = np.array(valid_combinations).reshape(-1, 1, len(K) + M)

    return valid_combinations_array

Dataset = np.load('Dataset.npz')

M = Dataset['M'].item()
T = Dataset['T'].item()
r = Dataset['r'].item()
l = Dataset['l'].item()
D = Dataset['D'].item()
P_U_dBm = Dataset['P_U_dBm'].item()
P_U_Watt = Dataset['P_U_Watt'].item()
P_W_dBm = Dataset['P_W_dBm']
P_W_Watt = Dataset['P_W_Watt']
B = Dataset['B'].item()
Noise_dBm = Dataset['Noise_dBm'].item()
Noise_Watt = Dataset['Noise_Watt'].item()
f_C_user = Dataset['f_C_user'].item()
f_C_worker = Dataset['f_C_worker']

K_possible = Dataset['K_possible']

Method = ['NMAB']

num_state = 1 + M + M + K_possible.shape[0]
num_action = 1


Num_Episode_Train = Dataset['Num_Episode_Train'].item()
Episode_Train = np.arange(Num_Episode_Train,dtype=int)

Channel_Gain_Train = Dataset['Channel_Gain_Train']

Lambda_Map_1 = Dataset['Lambda_Map_1_Train']
Lambda_Mask = Dataset['Lambda_Mask_Train']
Lambda_Encode = Dataset['Lambda_Encode_Train']
Lambda_Comp = Dataset['Lambda_Comp_Train']
Lambda_Decode = Dataset['Lambda_Decode_Train']
Lambda_Map_2 = Dataset['Lambda_Map_2_Train']
Lambda_Final = Dataset['Lambda_Final_Train']
Lambda_Multicast = Dataset['Lambda_Multicast_Train']
Lambda_Upload = Dataset['Lambda_Upload_Train']

batch_size_episode = 64
hidden_dim = 32
NMAB = DNN(num_state, hidden_dim, num_action)
NMAB_initial = copy.deepcopy(NMAB)
lambda_Train_NMAB = 1e-5
gamma_Train_NMAB = 1e-2
Z_Train_NMAB = lambda_Train_NMAB * torch.ones(count_parameters(NMAB),)
Solution_Train_NMAB, Latency_Train_NMAB = initialize_storage(Num_Episode_Train, M)
Loss_NMAB = []
experience_NMAB = []

for episode in range(Num_Episode_Train):

    channel_gain = Channel_Gain_Train[episode].flatten()

    channel_gain_state = channel_gain / channel_gain.max()

    combinations = generate_valid_combinations(M, K_possible, r, T)
    state = np.concatenate((np.tile(channel_gain_state, (combinations.shape[0], 1, 1)), combinations), axis=2)
    state = np.concatenate((np.tile(np.array([1]), (state.shape[0], 1, 1)), state), axis=2)
    state = np.vstack((np.concatenate((np.array([0]), channel_gain_state, np.zeros((M+K_possible.shape[0])))).reshape(1, 1, -1), state))

    action_NMAB = np.zeros((state.shape[0]))
    for idx in range(state.shape[0]):
        action_NMAB[idx] = NMAB(torch.as_tensor(state[idx], dtype=torch.float32)).detach().cpu().flatten()
        gradient = compute_gradient_single_output(NMAB, torch.as_tensor(state[idx], dtype=torch.float32))
        action_NMAB[idx] += torch.sqrt(torch.sum((gamma_Train_NMAB * 1.0 * gradient * gradient / Z_Train_NMAB / hidden_dim))).item()
    action = np.argmax(action_NMAB)
    if action != 0:
        k = np.argmax(state[action].flatten()[1+M:1+M+K_possible.shape[0]])
        K = K_possible[k]
        a = state[action].flatten()[1+M+K_possible.shape[0]:]
        if not np.all(np.sum(a) >= r*(K+T-1)+1):
            raise ValueError("Insufficient number of selected workers!")
        comp_overhead = {
            'Map_1': Lambda_Map_1[episode][k],
            'Mask': Lambda_Mask[episode][k],
            'Encode': Lambda_Encode[episode][k],
            'Comp': Lambda_Comp[episode][k],
            'Decode': Lambda_Decode[episode][k],
            'Map_2': Lambda_Map_2[episode][k],
            'Final': Lambda_Final[episode][k]
        }
        comm_overhead = {
            'Multicast': Lambda_Multicast[episode][k],
            'Upload': Lambda_Upload[episode][k]
        }
        eta_W = a / np.sum(a)
        objective_NMAB, system_latency_NMAB, comp_latency_NMAB, comm_latency_NMAB = get_latency(M, a, eta_W, channel_gain, comp_overhead, comm_overhead, P_U_Watt, P_W_Watt, B)
        Solution_Train_NMAB = update_solution(Solution_Train_NMAB, K, a, eta_W, episode)
        Latency_Train_NMAB = update_latency(Latency_Train_NMAB, objective_NMAB, system_latency_NMAB, comp_latency_NMAB, comm_latency_NMAB, episode)
    elif action == 0:
        objective_NMAB = system_latency_NMAB = get_overhead(l) / f_C_user
    reward_NMAB = get_reward(objective_NMAB)
    if episode < 10000:
        gradient = compute_gradient_single_output(NMAB, torch.as_tensor(state[action], dtype=torch.float32))
        Z_Train_NMAB += torch.square(gradient) / hidden_dim
        experience_NMAB.append([state[action], reward_NMAB])
        NMAB, loss = train(copy.deepcopy(NMAB), experience_NMAB)
        Loss_NMAB.append(loss)
    else:
        gradient = compute_gradient_single_output(NMAB, torch.as_tensor(state[action], dtype=torch.float32))
        Z_Train_NMAB += torch.square(gradient) / hidden_dim
        if episode % 100 == 0:
            experience_NMAB.append([state[action], reward_NMAB])
            NMAB, loss = train(copy.deepcopy(NMAB), experience_NMAB)
            Loss_NMAB.append(loss)

    if (episode+1) % 1 == 0:
        update_npz_file('NMAB', {'Num_Episode_Train':Num_Episode_Train})
        update_npz_file('NMAB',
                        {'Loss':Loss_NMAB,
                         'hidden_dim':hidden_dim,
                         'gamma_Train':gamma_Train_NMAB, 'lambda_Train':lambda_Train_NMAB})
        update_npz_file('NMAB',
                    {'Z_Train':Z_Train_NMAB})
        update_npz_file('NMAB', update_name(Solution_Train_NMAB, '_Train'))
        update_npz_file('NMAB', update_name(Latency_Train_NMAB, 't_', '_Train'))
        torch.save(NMAB.state_dict(), "NMAB")