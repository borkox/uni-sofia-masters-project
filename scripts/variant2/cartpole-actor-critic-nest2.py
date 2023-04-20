import math
import os, sys, getopt, glob
import gym
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.toy_text import FrozenLakeEnv
from pathlib import Path
import sklearn.preprocessing as scp


np.set_printoptions(precision=3)
clean_output = True
verbose = False
output_folder = "./outputs"
max_number_episodes = 60
opts, args = getopt.getopt(sys.argv[1:],"ho:c:n:v")
print (f"Script {sys.argv[0]} started with options: ", opts)
for opt, arg in opts:
    if opt in ("-h", "--help"):
        print ('USAGE:> cartpole-actor-critic-nest2.py OPTIONS')
        print ('OPTIONS: ')
        print ('   -o <output>')
        print ('       where <output> is output folder, default "./output"')
        print ('   -c <clean_output>')
        print ('       where <clean_output> is true or false, default true.')
        print ('   -n <num_episodes>')
        print ('       where <num_episodes> is max number of episodes, default 60.')
        print ('   -v')
        print ('       Verbose mode. Prints more debug messages.')
        sys.exit()
    elif opt in ("-o"):
        output_folder = arg
    elif opt in ("-n"):
        max_number_episodes = int(arg)
    elif opt in ("-c"):
        clean_output = 'true' == arg.lower()
    elif opt in ("-v"):
        verbose = True
    else:
        print("Unknown option:", opt)
        sys.exit()


print ('output_folder: ', output_folder)
print ('clean_output: ', clean_output)
print ('verbose: ', verbose)

import nest.voltage_trace

# Ensure folder with resources exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Clean output folder
if clean_output:
    [f.unlink() for f in Path(output_folder).glob("*") if f.is_file()]

# GAMMA for Sarsa TD(0)
GAMMA = 0.95
# number of episodes to run
NUM_EPISODES = max_number_episodes
# max steps per episode
MAX_STEPS = 200
# Saves scores to file evey SAVE_SCORES_STEPS steps
SAVE_SCORES_STEPS = 5
# score agent needs for environment to be solved
SOLVED_HISTORY_SCORES_LEN = 10
SOLVED_MEAN_SCORE = 195
# current time while it runs
current_time = 0
# STEP is milliseconds to wait WTA to become functional
STEP = 120
# Learn time is when WTA is still active and dopamine is activated
LEARN_TIME = 20
# REST_TIME is milliseconds to run rest for WTA and perform dopamine STDP
REST_TIME = 40
# Noise constants
NOISE_DA_NEURONS_WEIGHT = 0.01
NOISE_ALL_STATES_WEIGHT = 0.01
NOISE_RATE = 65000.
CRITIC_NOISE_RATE = 65500.
REWARD_STIMULUS_RATE = 65000.
STIMULUS_RATE = 65000.
WTA_NOISE_RATE = 500.
INPUT_TRANSFORMED_DIM = 8

# ================================================
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

# ================= Environment ==================
# Make environment
env = gym.make('CartPole-v1')

num_actions = 2
possible_actions = [0, 1]
possible_actions_str = ["LEFT", "RIGHT"]
# ================================================

NUM_STATE_NEURONS = 20
NUM_WTA_NEURONS = 50
# WEIGHT_SCALING = 100 / NUM_STATE_NEURONS
REWARD_SCALING = 2.5 * 20 / NUM_STATE_NEURONS
DA_NEURONS = 150

rank = nest.Rank()
size = nest.NumProcesses()
seed = np.random.randint(0, 1000000)
num_threads = 1
nest.SetKernelStatus({"local_num_threads": num_threads})
nest.SetKernelStatus({"rng_seed": seed})
tau_pre = 20.
nest.SetDefaults("iaf_psc_alpha", {"tau_minus": tau_pre})

# Create states
states = []
all_states = None
for i in range(INPUT_TRANSFORMED_DIM):
    state_group = nest.Create('iaf_psc_alpha', NUM_STATE_NEURONS)
    states.append(state_group)
    if all_states is None:
        all_states = state_group
    else:
        all_states = all_states + state_group

if verbose:
    print('all_states: ', all_states)

# Create actions
actions = []
all_actions = None
for i in range(num_actions):
    action_group = nest.Create('iaf_psc_alpha', NUM_WTA_NEURONS)
    actions.append(action_group)
    if all_actions is None:
        all_actions = action_group
    else:
        all_actions = all_actions + action_group
if verbose:
  print("Len (all_actions)=",all_actions)
# Create WTA circuit
wta_ex_weights = 10.5
wta_inh_weights = -2.6
wta_ex_inh_weights = 2.8
wta_noise_weights = 2.1

wta_inh_neurons = nest.Create('iaf_psc_alpha', NUM_WTA_NEURONS)

for i in range(len(actions)):
    nest.Connect(actions[i], actions[i], 'all_to_all', {'weight': wta_ex_weights})
    nest.Connect(actions[i], wta_inh_neurons, 'all_to_all', {'weight': wta_ex_inh_weights})

nest.Connect(wta_inh_neurons, all_actions, 'all_to_all', {'weight': wta_inh_weights})

wta_noise = nest.Create('poisson_generator', 10, {'rate': WTA_NOISE_RATE})
nest.Connect(wta_noise, all_actions, 'all_to_all', {'weight': wta_noise_weights})
nest.Connect(wta_noise, wta_inh_neurons, 'all_to_all', {'weight': wta_noise_weights * 0.9})

# Create stimulus
stimulus = nest.Create('poisson_generator', 1, {'rate': STIMULUS_RATE})
nest.Connect(stimulus, all_states, 'all_to_all', {'weight': 0.})

# Here, we are implementing the dopaminergic nueron pool, volume transmitter and dopamin-modulated synapse between states and actions

# Create DA pool
DA_neurons = nest.Create('iaf_psc_alpha', DA_NEURONS)
# vol_trans = nest.Create('volume_transmitter', 1, {'deliver_interval': 10})
vol_trans = nest.Create('volume_transmitter', 1)
nest.Connect(DA_neurons, vol_trans, 'all_to_all')

# Create reward stimulus
reward_stimulus = nest.Create('poisson_generator', 1, {'rate': REWARD_STIMULUS_RATE})
nest.Connect(reward_stimulus, DA_neurons, 'all_to_all', {'weight': 0.})

tau_c = 200.0  # Time constant of eligibility trace
tau_n = 5.0  # Time constant of dopaminergic trace
tau_plus = 20.

# Connect states to actions
nest.CopyModel('stdp_dopamine_synapse', 'dopa_synapse', {
    'vt': vol_trans.get('global_id'), 'A_plus': 0.05, 'A_minus': 0.05, "tau_plus": tau_plus,
    'Wmin': -100., 'Wmax': 100., 'b': 0., 'tau_n': tau_n, 'tau_c': tau_c})

nest.Connect(all_states, all_actions, 'all_to_all', {'synapse_model': 'dopa_synapse', 'weight': 0.0})

critic = nest.Create('iaf_psc_alpha', 50)
nest.Connect(all_states, critic, 'all_to_all', {'synapse_model': 'dopa_synapse', 'weight': 0.0})
nest.Connect(critic, DA_neurons, 'all_to_all', {'weight': -200., 'delay': STEP+LEARN_TIME+REST_TIME})
nest.Connect(critic, DA_neurons, 'all_to_all', {'weight': GAMMA * 200., 'delay': 1.})

critic_noise = nest.Create('poisson_generator', 1, {'rate': CRITIC_NOISE_RATE})
nest.Connect(critic_noise, critic)

# Create spike detector
sd_wta = nest.Create('spike_recorder')
nest.Connect(all_actions, sd_wta)
#nest.Connect(wta_inh_neurons, sd_wta)
sd_actions = nest.Create('spike_recorder', num_actions)
for i in range(len(actions)):
    nest.Connect(actions[i], sd_actions[i])
sd_states = nest.Create('spike_recorder')
nest.Connect(all_states, sd_states)
sd_DA = nest.Create('spike_recorder', 1)
nest.Connect(DA_neurons, sd_DA, 'all_to_all')
sd_critic = nest.Create('spike_recorder', 1)
nest.Connect(critic, sd_critic, 'all_to_all')

# Create noise
noise = nest.Create('poisson_generator', 1, {'rate': NOISE_RATE})
nest.Connect(noise, all_states, 'all_to_all', {'weight': NOISE_ALL_STATES_WEIGHT})
nest.Connect(noise, DA_neurons, 'all_to_all', {'weight': NOISE_DA_NEURONS_WEIGHT})


# Init network
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space.n}")

scaler = scp.MinMaxScaler(feature_range=(0.1, 1), copy=True, clip=True)
# See https://www.gymlibrary.dev/environments/classic_control/cart_pole/#observation-space
scaler.fit([[0, 0, 0, 0, 0, 0, 0, 0], [+1.5, +1.5, +1.5, +1.5, +0.13, +0.13, +2.1, +2.1]])


def transform_state(s):
    new_transformed_state = np.array([
        abs(s[0]) if s[0] > 0 else 0,
        abs(s[0]) if s[0] < 0 else 0,
        abs(s[1]) if s[1] > 0 else 0,
        abs(s[1]) if s[1] < 0 else 0,
        abs(s[2]) if s[2] > 0 else 0,
        abs(s[2]) if s[2] < 0 else 0,
        abs(s[3]) if s[3] > 0 else 0,
        abs(s[3]) if s[3] < 0 else 0])
    new_transformed_state_scaled = scaler.transform(new_transformed_state.reshape(1, -1)).reshape(-1)
    transformed = (np.exp(new_transformed_state_scaled) * 1.5 - 1)
    if verbose:
      print("Converting state: ", s)
      print("  ==> ", new_transformed_state,)
      print("  ==> ", new_transformed_state_scaled)
      print("  ==> ", transformed)
    return transformed

# track scores
scores = []

# track recent scores
recent_scores = deque(maxlen=SOLVED_HISTORY_SCORES_LEN)
prev_spikes = 0
# run episodes
for episode in range(NUM_EPISODES):
    nest.SetStatus(sd_actions, {"n_events": 0})
    nest.SetStatus(sd_wta, {"n_events": 0})
    nest.SetStatus(sd_states, {"n_events": 0})
    nest.SetStatus(sd_DA, {"n_events": 0})
    nest.SetStatus(sd_critic, {"n_events": 0})

    # init variables
    state = env.reset()
    print("STATE:+>>>>", state)
    done = False
    score = 0
    reward = 0
    step = 0
    # run episode, update online
    for _ in range(MAX_STEPS):

        # Supress learning
        nest.SetStatus(DA_neurons, {'I_e': -1000.})

        # Apply environment state as weights from "stimulus" to "states[i]"
        state_tr = transform_state(state)
        for i in range(INPUT_TRANSFORMED_DIM):
            nest.SetStatus(nest.GetConnections(stimulus, states[i]), {'weight': state_tr[i]})
        nest.SetStatus(wta_noise, {'rate': 3000.})

        env.render()
        nest.Simulate(STEP)

        max_rate = -1
        chosen_action = -1
        rates = []
        for i in range(len(sd_actions)):
            rate = len([e for e in nest.GetStatus(sd_actions[i], keys='events')[0]['times'] if
                        e > current_time])  # calc the \"firerate\" of each actor population
            rates.append(rate)
            if rate > max_rate:
                max_rate = rate  # the population with the hightes rate wins
                chosen_action = i

        current_time += STEP
        print("chose action:", possible_actions[chosen_action], " ", possible_actions_str[chosen_action], " at step ",
              step, " rates: ", rates)

        action = possible_actions[chosen_action]
        new_state, reward, done, _ = env.step(action)

        # Stop stimulus from environment
        nest.SetStatus(nest.GetConnections(stimulus, all_states), {'weight': 0.})

        # apply reward
        reward = max(10 * math.cos(17 * new_state[2]), 0)
        print("Scaled reward:", float(reward) * REWARD_SCALING)
        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': float(reward) * REWARD_SCALING})

        # learn time
        if reward > 0 or not done:
            print("Learn time")
            # Enable learning
            nest.SetStatus(DA_neurons, {'I_e': 0.})
            nest.Simulate(LEARN_TIME)
            current_time += LEARN_TIME
            # Supress learning
            nest.SetStatus(DA_neurons, {'I_e': -1000.})
        else:
            nest.Simulate(LEARN_TIME)
            current_time += LEARN_TIME
            print("No learn on this step.")

        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': 0.0})

        nest.SetStatus(wta_noise, {'rate': 0.})
        # refactory time
        nest.Simulate(REST_TIME)
        current_time += REST_TIME

        score += reward

        # if terminal state, next state val is 0
        if done:
            print(f"Episode {episode} finished after {step} timesteps and reward {score}")
            break

        # move into new state, discount I
        state = new_state
        step = step + 1

    # append episode score
    scores.append(score)
    recent_scores.append(score)

    # early stopping if we meet solved score goal
    if np.array(recent_scores).mean() >= SOLVED_MEAN_SCORE \
            and reward > SOLVED_MEAN_SCORE: # We want spikes to be snapshot when actually episode is solved
        print("SOLVED")
        break
    else:
        print('Mean score: ', np.array(recent_scores).mean())
    if len(scores) % SAVE_SCORES_STEPS == 0:
        print("Save scores")
        np.savetxt(output_folder + '/scores.txt', scores, delimiter=',')


# if reward > 0:
    #     break
np.savetxt(output_folder + '/scores.txt', scores, delimiter=',')

print("====== all_states === all_actions ===")
print(nest.GetConnections(all_states, all_actions))

nest.raster_plot.from_device(sd_wta, hist=True, title="sd_wta")
plt.savefig(f'{output_folder}/sd_wta.png', format='png')
nest.raster_plot.from_device(sd_states, hist=True, title="sd_states")
plt.savefig(f'{output_folder}/sd_states.png', format='png')
nest.raster_plot.from_device(sd_DA, hist=True, title="sd_DA")
plt.savefig(f'{output_folder}/sd_DA.png', format='png')
nest.raster_plot.from_device(sd_critic, hist=True, title="sd_critic")
plt.savefig(f'{output_folder}/sd_critic.png', format='png')

# Print scores
os.system(f"python ./plot_scores.py -i {output_folder}/scores.txt -o {output_folder}/final_scores.png")