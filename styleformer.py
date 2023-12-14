# coding=utf-8

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

adequacy_tag = 'prithivida/parrot_adequacy_model'
ftc_tag = "prithivida/formal_to_informal_styletransfer"
ctf_tag = "prithivida/informal_to_formal_styletransfer"
atp_tag = "prithivida/active_to_passive_styletransfer"
pta_tag = "prithivida/passive_to_active_styletransfer"


class Adequacy:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(adequacy_tag)
        self.model = AutoModelForSequenceClassification.from_pretrained(adequacy_tag, torch_dtype=torch.float32)

    def filter(self, input_phrase, para_phrases, threshold, device="cpu"):
        top_phrases = []
        for phrase in para_phrases:
            x = self.tokenizer(input_phrase, phrase, return_tensors='pt', max_length=128, truncation=True)
            self.model = self.model.to(device)

            logits = self.model(**x).logits
            probs = logits.softmax(dim=1)
            prob_label = probs[:, 1]

            score = prob_label.item()
            if score >= threshold:
                top_phrases.append(phrase)

        return top_phrases

    def score(self, input_phrase, para_phrases, threshold, device="cpu"):
        scores = {}
        for phrase in para_phrases:
            x = self.tokenizer(input_phrase, phrase, return_tensors='pt', max_length=128, truncation=True)
            x = x.to(device)
            self.model = self.model.to(device)

            logits = self.model(**x).logits
            probs = logits.softmax(dim=1)
            prob_label = probs[:, 1]

            score = prob_label.item()
            if score >= threshold:
                scores[phrase] = score

        return scores


class StyleFormer:
    def __init__(self, style=0):
        self.style = style

    def transfer(self, sen, filter=0.95, candidates=5):
        device = "cpu"
        if self.style == 0:
            output = self._casual_to_formal(sen, device, filter, candidates)
            return output

        elif self.style == 1:
            output = self._formal_to_casual(sen, device, filter, candidates)
            return output

        elif self.style == 2:
            output = self._active_to_passive(sen, device)
            return output

        elif self.style == 3:
            output = self._passive_to_active(sen, device)
            return output

    def _formal_to_casual(self, sen, device, filter, candidates):
        self.ftc_tokenizer = AutoTokenizer.from_pretrained(ftc_tag)
        self.ftc_model = AutoModelForSeq2SeqLM.from_pretrained(ftc_tag, torch_dtype=torch.float32)

        ftc_prefix = "transfer Formal to Casual: "
        src_sen = sen
        sen = ftc_prefix + sen
        input_ids = self.ftc_tokenizer.encode(sen, return_tensors='pt')

        self.ftc_model = self.ftc_model.to(device)
        input_ids = input_ids.to(device)

        preds = self.ftc_model.generate(
            input_ids,
            do_sample=True,
            max_length=128,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=candidates
        )

        gen_sen = set()
        for pred in preds:
            gen_sen.add(self.ftc_tokenizer.decode(pred, skip_special_tokens=True).strip())

        self.adequacy = Adequacy()
        phrases = self.adequacy.score(src_sen, list(gen_sen), filter, device)
        ranking = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        if len(ranking) > 0:
            return ranking[0][0]
        else:
            return None

    def _casual_to_formal(self, sen, device, filter, candidates):
        self.ctf_tokenizer = AutoTokenizer.from_pretrained(ctf_tag)
        self.ctf_model = AutoModelForSeq2SeqLM.from_pretrained(ctf_tag, torch_dtype=torch.float32)

        ctf_prefix = "transfer Casual to Formal: "
        src_sen = sen
        sen = ctf_prefix + sen
        input_ids = self.ctf_tokenizer.encode(sen, return_tensors='pt')

        self.ctf_model = self.ctf_model.to(device)
        input_ids = input_ids.to(device)

        preds = self.ctf_model.generate(
            input_ids,
            do_sample=True,
            max_length=128,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=candidates
        )

        gen_sen = set()
        for pred in preds:
            gen_sen.add(self.ctf_tokenizer.decode(pred, skip_special_tokens=True).strip())

        self.adequacy = Adequacy()
        phrases = self.adequacy.score(src_sen, list(gen_sen), filter, device)
        ranking = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        if len(ranking) > 0:
            return ranking[0][0]
        else:
            return None

    def _active_to_passive(self, sen, device):
        self.atp_tokenizer = AutoTokenizer.from_pretrained(atp_tag)
        self.atp_model = AutoModelForSeq2SeqLM.from_pretrained(atp_tag, torch_dtype=torch.float32)

        atp_prefix = "transfer Active to Passive: "
        sen = atp_prefix + sen
        input_ids = self.atp_tokenizer.encode(sen, return_tensors='pt')

        self.atp_model = self.atp_model.to(device)
        input_ids = input_ids.to(device)

        preds = self.atp_model.generate(
            input_ids,
            do_sample=True,
            max_length=128,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1
        )

        return self.atp_tokenizer.decode(preds[0], skip_special_tokens=True).strip()

    def _passive_to_active(self, sen, device):
        self.pta_tokenizer = AutoTokenizer.from_pretrained(pta_tag)
        self.pta_model = AutoModelForSeq2SeqLM.from_pretrained(pta_tag, torch_dtype=torch.float32)

        pta_prefix = "transfer Passive to Active: "
        sen = pta_prefix + sen
        input_ids = self.pta_tokenizer.encode(sen, return_tensors='pt')

        self.pta_model = self.pta_model.to(device)
        input_ids = input_ids.to(device)

        preds = self.pta_model.generate(
            input_ids,
            do_sample=True,
            max_length=128,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1
        )

        return self.pta_tokenizer.decode(preds[0], skip_special_tokens=True).strip()
