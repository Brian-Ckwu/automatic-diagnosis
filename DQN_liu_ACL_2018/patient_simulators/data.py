from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class PatientSimDataset(Dataset):
    # Make the dataset for training or evaluating the patient simulator from a split (train / test) of patient states
    def __init__(self, patients: List[dict], tokenizer: BertTokenizerFast, sampler_mode: str = "cn_1", pol_mode: str = "all_wo_pol"): # pol_mode in ["pos", "all_wo_pol", "all_w_pol"]
        self.patients = patients
        self.tokenizer = tokenizer
        self.sampler_mode = sampler_mode
        self.pol_mode = pol_mode

        if self.sampler_mode == "cn_1":
            self.findings_sampler = self.naive_findings_sampler
        # merge all findings
        self.findings_l = [ # chief complaint + implicit findings
            list(patient["chief_complaint"].items()) + \
            list(patient["clinical_findings"].items()) \
            for patient in self.patients
        ]
        # make samples from patients
        self.samples = list()
        for patient, findings in zip(self.patients, self.findings_l):
            p_samples = self.make_samples(patient, findings)
            self.samples += p_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
    
    def make_samples(self, patient: dict, findings: list) -> List[dict]: # {observed_info, requested_finding, label}
        samples = list()
        findings_sampler = self.findings_sampler(findings) # NOTE: can also use other samplers
        for obs_fs, req_f in findings_sampler:
            sample = {
                "observed_info": {
                    "diagnosis": patient["diagnosis"],
                    "findings": obs_fs # TODO: add more basic information
                },
                "requested_fdn": req_f[0],
                "polarity": req_f[1]
            }
            samples.append(sample)

        return samples

    # Data collator for dataloaders
    def collate_fn(self, samples: list):
        batch = list()
        labels = list()

        for sample in samples:
            dx = sample["observed_info"]["diagnosis"]

            fnp = sample["observed_info"]["findings"]
            if fnp:
                obs_findings, pols = zip(*fnp)
            else:
                obs_findings = list()
            # TODO: handle tokens for polarity embeddings
            req_fdn = sample["requested_fdn"]
            label = int(sample["polarity"])

            obs_info = [dx] + list(obs_findings)
            obs_req_info = obs_info + [f"{self.tokenizer.sep_token} {req_fdn}"]
            batch.append(' '.join(obs_req_info))
            labels.append(label)

        batch_encs = self.tokenizer(batch, padding=True, return_tensors="pt")
        # Create token type ids mask
        segb_locs = (batch_encs.input_ids == self.tokenizer.sep_token_id).int().argmax(dim=1) + 1
        for i in range(len(segb_locs)):
            batch_encs.token_type_ids[i].index_fill_(dim=-1, index=torch.arange(segb_locs[i], batch_encs.token_type_ids.size(1)), value=1)        
        assert (batch_encs.token_type_ids.argmax(dim=1) == segb_locs).all().item() # check for token type ids' correctness

        labels = torch.LongTensor(labels)

        return batch_encs, labels

    # Sample findings by C(n, n-1)
    @staticmethod
    def naive_findings_sampler(findings: List[Tuple[str]]) -> Tuple[List[Tuple[str]], Tuple[str]]:
        for i in range(len(findings)):
            obs_fs = findings[:i] + findings[i + 1:]
            req_f = findings[i]
            yield obs_fs, req_f

if __name__ == "__main__":
    # For testing
    from utils import load_json
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    patients = load_json(file="/nfs/nas-7.1/ckwu/automatic_diagnosis/DQN_liu_ACL_2018/data/patient_states.json")
    train_patients = patients["train"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    train_set = PatientSimDataset(patients=train_patients, tokenizer=tokenizer, sampler_mode="cn_1", pol_mode="all_wo_pol")
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, pin_memory=True, collate_fn=train_set.collate_fn)

    for batch in train_loader:
        pass