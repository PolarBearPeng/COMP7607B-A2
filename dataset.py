import json
import os

import torch
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(file_path)

    def load_data(self, path):
        samples = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def get_sources(self, samples):
        raise NotImplementedError("get_sources method is not implemented for PretrainDataset")

    def get_references(self, samples):
        raise NotImplementedError("get_references method is not implemented for PretrainDataset")

    def __getitem__(self, index):
        sample = self.samples[index]

        # Build input text
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        # text = str(sample['text'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = input_ids != self.tokenizer.pad_token_id

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

    def __len__(self):
        return len(self.samples)


class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(file_path)
        self.bos_id = tokenizer("<s>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("</s>\n", add_special_tokens=False).input_ids
        self.prompt_length = 65

    def load_data(self, file_path):
        samples = []
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """Build dialogue in ChatML format"""
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def get_sources(self, samples):
        sources = []
        for sample in samples:
            conversations = sample["conversations"]
            source = conversations[0]["content"][self.prompt_length :].strip()
            sources.append(source)
        return sources

    def get_references(self, samples):
        references = []
        for sample in samples:
            conversations = sample["conversations"]
            reference = conversations[1]["content"].strip()
            references.append(reference)
        return references

    def extract_messages(self, sample):
        return [sample["conversations"][0]]

    def __getitem__(self, index):
        sample = self.samples[index]
        # Build dialogue prompt
        prompt = self._create_chat_prompt(sample["conversations"])
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # Generate dynamic loss mask
        loss_mask = self._generate_loss_mask(input_ids)

        # Build training data
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask

    def __len__(self):
        return len(self.samples)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer("<s>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("</s>\n", add_special_tokens=False).input_ids
        self.samples = self.load_data(file_path)
        self.prompt_length = 65

    def load_data(self, file_path):
        samples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                samples.append(obj)
        return samples

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def get_sources(self, samples):
        sources = []
        for sample in samples:
            chosen = sample["chosen"]
            source = chosen[0]["content"][self.prompt_length :].strip()
            sources.append(source)
        return sources

    def get_references(self, samples):
        references = []
        for sample in samples:
            chosen = sample["chosen"]
            reference = chosen[1]["content"].strip()
            references.append(reference)
        return references

    def extract_messages(self, sample):
        return [sample["chosen"][0]]

    def __getitem__(self, index):
        item = self.samples[index]
        chosen = item["chosen"]  # A list containing multiple {role, content} pairs
        rejected = item["rejected"]  # Same as above
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }

    def __len__(self):
        return len(self.samples)
