from test_models import GATNNQ, My_MLP
# from test_theory_plot import get_standard_plot
# from torch.utils.tensorboard import SummaryWriter
import pickle
import torch.nn.functional as F
from env_test import EnvTest
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

lagrange_factor = 2.0
lagrange_factor = float(lagrange_factor)
# use_baseline = False
use_baseline = True
episode_round = 100

train_lam = False
# train_lam = True

seed = 1
torch.manual_seed(0)
np.random.seed(0)
random.seed(seed)


def list_to_tensor(list_):
    if not isinstance(list_, list):
        list_ = [list_]
    temp = torch.tensor(list_)
    temp = temp.view(1, -1, 1)
    temp = torch.tensor(temp, dtype=torch.float32)
    return temp


def get_action(_my_model, s):
    # print("aaa ", my_model(list_to_tensor(s), list_to_tensor(0)))
    # print("bbb ", my_model(list_to_tensor(s), list_to_tensor(1)))
    if random.random() > 0.1:
        action = 0 if (_my_model(list_to_tensor(s), list_to_tensor(0)) >= _my_model(list_to_tensor(s),
                                                                                   list_to_tensor(1))) else 1
    else:
        action = 0 if (_my_model(list_to_tensor(s), list_to_tensor(0)) < _my_model(list_to_tensor(s),
                                                                                   list_to_tensor(1))) else 1
    return action
    # return 1


def get_best_action(_my_model, s):
    # print("aaa ", my_model(list_to_tensor(s), list_to_tensor(0)))
    # print("bbb ", my_model(list_to_tensor(s), list_to_tensor(1)))
    action = 0 if (
            _my_model(list_to_tensor(s), list_to_tensor(0)) >= _my_model(list_to_tensor(s), list_to_tensor(1))) else 1
    return action
    # return 1


if __name__ == '__main__':
    # critic_local_mlp = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
    #                                            key_to_use="Critic")

    ###
    # writer_name = "test_writer" + str(use_baseline) + str(episode_round)
    # test_writer = SummaryWriter(comment=writer_name, filename_suffix="_tzm")
    ###

    # record_y_zero = []
    # record_y_one = []
    # record_y_ = []

    record_reward = []
    _record_reward = []

    env = EnvTest()
    env2 = EnvTest()

    ##### models ####################################################################################
    state_size = 2
    action_size = 1
    adj = torch.FloatTensor(np.ones((state_size, state_size))).unsqueeze(0)
    critic_local_gat = GATNNQ(input_state_dim=state_size, input_action_dim=action_size, output_dim=1,
                              mask_dim=state_size, emb_dim=4, window_size=1,
                              nheads=1, adj_base=adj)

    mlp_model = My_MLP(input_dim=state_size + action_size, output_dim=1)

    #############################################################################################################

    if use_baseline:
        my_model = mlp_model
    else:
        my_model = critic_local_gat

    lr_ = 1e-3
    optimizer = torch.optim.Adam(my_model.parameters(), lr=lr_, eps=1e-4)

    #########################################################################
    lam = torch.tensor(lagrange_factor, requires_grad=True)
    lam_optimiser = torch.optim.Adam([lam], lr=1e-2)
    #########################################################################

    for ep_id in range(episode_round):
        print("ep_id is ", ep_id)
        env.reset()
        total_reward = 0
        for step_id in range(env.state_number):
            env.lagrange_factor = float(lam.detach().cpu().numpy())
            print("ssss ", env.lagrange_factor)
            s = env.state
            action = get_action(my_model, s)
            # print("action is ", action)
            s_, reward, done_info = env.step(action)
            # print("reward is ", reward)
            total_reward += reward
            if (1 > 0):
                q_next = my_model(list_to_tensor(s_), list_to_tensor(get_best_action(my_model, s_)))
                next_q_value = reward + (1 - done_info) * q_next

                loss = F.mse_loss(next_q_value, my_model(list_to_tensor(s), list_to_tensor(action)))

                optimizer.zero_grad()  # reset gradients to 0
                loss.backward()  # this calculates the gradients
                optimizer.step()  # this applies the gradients

                #############################################
                if train_lam:
                    violation = torch.tensor(float(env.penalty), requires_grad=False)
                    # print("violation is ", violation)
                    log_lam = torch.nn.functional.softplus(lam)
                    lambda_loss = violation * log_lam
                    # print(lambda_loss)

                    lam_optimiser.zero_grad()  # reset gradients to 0
                    lambda_loss.backward()  # this calculates the gradients
                    lam_optimiser.step()  # this applies the gradients
                #############################################

            if done_info:
                break

        # print(total_reward)
        record_reward.append(total_reward)

        # for name, param in my_model.named_parameters():
        #     test_writer.add_histogram(name, param.clone().cpu().data.numpy(), ep_id)
        #     test_writer.add_histogram(name + "/grad", param.grad.clone().cpu().data.numpy(), ep_id)

        _total_reward = 0
        for _ in range(env2.state_number):
            env2.lagrange_factor = lagrange_factor
            _s = env2.state
            _action = get_best_action(my_model, _s)
            _s_, _reward, _done_info = env2.step(_action)
            # print("action is ", _action)
            # print("reward is ", _reward)
            _total_reward += _reward
            # print("total reward is ", _reward)
            if _done_info:
                break
        env2.reset()
        _record_reward.append(_total_reward)
        # print("test_reward is ", _total_reward)

    plt.plot(record_reward)
    plt.plot(_record_reward, color="red")
    plt.show()

