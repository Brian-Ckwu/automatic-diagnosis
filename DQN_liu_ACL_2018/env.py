import random
from argparse import Namespace

import numpy as np

from gym import Env, spaces
from stable_baselines3.common.env_checker import check_env

from utils import load_json

class PatientSim(Env):
    metadata = {"render_modes": [None, "dict", "vector"]}

    def __init__(self, env_args: Namespace):
        super().__init__()
        self.render_mode = None # prevent rendering automatically
        
        # Load data
        self.patient_states = load_json(env_args.patient_states_path)[env_args.split]
        self.cf2idx = load_json(env_args.cf2idx_path)
        self.dx2idx = load_json(env_args.dx2idx_path)
        self.cf_encoding = env_args.cf_encoding

        self.shuffle_patients()
        self.idx2cf = self.get_reverse_dict(d=self.cf2idx)
        self.idx2dx = self.get_reverse_dict(d=self.dx2idx)

        # Instance attributes
        self.seed = env_args.seed
        self.cf_num = len(self.cf2idx)
        self.dx_num = len(self.dx2idx)
        self.max_turns = env_args.max_turns
        self.step_penalty = env_args.step_penalty
        self.dx_reward = env_args.dx_reward
        self.dx_penalty = env_args.dx_penalty

        # Variables
        self.cur_turns = 0
        self.cur_full_state: dict = dict()
        self.cur_obs_state: dict = dict()
        self.cur_obs_vec: np.ndarray = None

        # Possible observations
        self.observation_space = spaces.Box(
            low=min(self.cf_encoding.values()), 
            high=max(self.cf_encoding.values()),
            shape=(self.cf_num,),
            dtype=int
        )

        # Possible actions
        self.action_space = spaces.Discrete(
            n=self.cf_num + self.dx_num, 
            seed=self.seed
        )

    # Initiate a new episode by setting initial "full patient state" and "observed patient state"
    def reset(self, options: dict = {"patient_idx": 0}, return_info: bool = False):
        self.clear()
        # Set the patient state by index
        pid = options["patient_idx"]
        self.cur_full_state = self.patient_states[pid].copy()
        self.cur_obs_state = self.cur_full_state.copy()
        self.cur_obs_state["diagnosis"] = None
        self.cur_obs_state["clinical_findings"] = dict() # only "chief complaint" is initially observable

        # Return values
        observation = self._get_obs_vec()
        info = dict()
        info["cur_full_state"] = self.cur_full_state.copy()

        return (observation, info) if return_info else observation

    def step(self, action):
        self.cur_turns += 1
        done = self.cur_turns >= self.max_turns
        # Determine whether the action is "request a finding" or "inform the diagnosis"
        # request a finding
        if action < self.cf_num:
            rf = self.idx2cf[action]
            # Check if the clinical finding is already documented or asked
            if self._rf_is_in_obs(rf):
                # TODO: do something (e.g. give penalty)
                pass
            # If not asked, check if the finding is in the full state
            elif self._rf_is_in_full(rf):
                pol = self.cur_full_state["clinical_findings"][rf]
                pol_s = "positive" if pol else "negative"
                self.cur_obs_state["clinical_findings"][rf] = pol
                self.cur_obs_vec[action] = self.cf_encoding[pol_s]
            else:
                pol_s = "not_sure"
                self.cur_obs_state["clinical_findings"][rf] = pol_s
                self.cur_obs_vec[action] = self.cf_encoding[pol_s]
            reward = self.step_penalty

        # inform a diagnosis
        else:
            idx = action - self.cf_num
            dx = self.idx2dx[idx]
            # Check if the diagnosis is correct
            if dx == self.cur_full_state["diagnosis"]:
                dx_correct = True
            else:
                dx_correct = False

            reward = self.dx_reward if dx_correct else self.dx_penalty
            done = True

        observation = self._get_obs_vec()
        info = {}
        return observation, reward, done, info # information of acc, turns, match rate, ...

    def render(self, mode: str):
        assert mode in self.metadata["render_modes"]
        if mode == "dict":
            return self.cur_obs_state
        elif mode == "vector":
            return self._get_obs_vec()
    
    def clear(self):
        self.cur_turns = 0
        self.cur_full_state: dict = dict()
        self.cur_obs_state: dict = dict()
        self.cur_obs_vec: np.ndarray = None

    def shuffle_patients(self) -> None:
        random.shuffle(self.patient_states)

    def _get_obs_vec(self):
        if self.cur_obs_vec is None:
            cc = self.cur_obs_state.get("chief_complaint", {})
            cf = self.cur_obs_state.get("clinical_findings", {})

            obs_vec = np.zeros(shape=(self.cf_num,), dtype=int)
            for f2p in [cc, cf]:
                for f, p in f2p.items():
                    f_idx = self.cf2idx[f]
                    if p == True:
                        f_pol = "positive"
                    elif p == False:
                        f_pol = "negative"
                    else:
                        f_pol = "not_sure"
                    obs_vec[f_idx] = self.cf_encoding[f_pol]

            self.cur_obs_vec = obs_vec

        return self.cur_obs_vec
    
    def _rf_is_in_obs(self, rf: str) -> bool:
        return (rf in self.cur_obs_state.get("chief_complaint", {})) or (rf in self.cur_obs_state.get("clinical_findings", {}))
    
    def _rf_is_in_full(self, rf: str) -> bool:
        return rf in self.cur_full_state["clinical_findings"]


    @staticmethod
    def get_reverse_dict(d: dict) -> dict:
        return {v: k for k, v in d.items()}

if __name__ == "__main__":
    # chech env
    env_args = Namespace(**load_json(file="./env_args.json"))
    env = PatientSim(env_args)
    check_env(env)