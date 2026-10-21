import math
import os
import json
import logging
from collections import defaultdict
import numpy as np
from copy import deepcopy
from higher.patch import monkeypatch as make_functional
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from src.models.modeling_bert import BertForMaskedLM, BertEncoder, EXBertForMaskedLM, DualBertForMaskedLM
from src.data.data_module import dataset_mapping


class Args:
    def __init__(self, **entries) -> None:
        self.__dict__.update(entries)


logger = logging.getLogger(__name__)


class Adapter(nn.Module):
    def __init__(self, args, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)
        self.init_weights()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()


class PretrainedModel(nn.Module):
    def __init__(self, model_name_or_path, kge_model_type):
        super(PretrainedModel, self).__init__()
        pretrain_model = BertForMaskedLM.from_pretrained(model_name_or_path, output_hidden_states=True)
        self.model = pretrain_model.bert
        self.cls = pretrain_model.cls
        self.config = pretrain_model.config
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             head_mask=head_mask)

        return outputs  # (loss), logits, (hidden_states), (attentions) bert: logits, (hidden_states), (attetions)


class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config, n_rel):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = self.args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float = 0.1
            hidden_dropout_prob: float = 0.1
            hidden_size: int = 768
            initializer_range: float = 0.02
            intermediate_size: int = 3072
            layer_norm_eps: float = 1e-05
            max_position_embeddings: int = 514
            num_attention_heads: int = 12
            num_hidden_layers: int = self.args.adapter_transformer_layers
            num_labels: int = 2
            output_attentions: bool = False
            output_hidden_states: bool = False
            torchscript: bool = False
            type_vocab_size: int = 1
            vocab_size: int = 50265
            chunk_size_feed_forward: int = self.config.chunk_size_feed_forward
            add_cross_attention: bool = self.config.add_cross_attention

        self.adapter_skip_layers = self.args.adapter_skip_layers
        self.num_labels = n_rel
        # self.config.output_hidden_states=True
        self.adapter_list = args.adapter_list
        # self.adapter_list =[int(i) for i in self.adapter_list]
        self.adapter_num = len(self.adapter_list)
        # self.adapter = Adapter(args, AdapterConfig)

        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

        self.com_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.out_proj = None

    def forward(self, pretrained_model_outputs, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):

        outputs = pretrained_model_outputs
        sequence_output = outputs[0]  # last hidden states
        # pooler_output = outputs[1]
        hidden_states = outputs[1]  # (8, 64, 768)
        num = len(hidden_states)

        hidden_states_last = torch.zeros(sequence_output.size()).to(self.args.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1:  # if adapter_skip_layers>=1, skip connection
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + adapter_hidden_states[
                        int(adapter_hidden_states_count / self.adapter_skip_layers)]

        ##### drop below parameters when doing downstream tasks
        com_features = self.com_dense(torch.cat([sequence_output, hidden_states_last], dim=2))

        # subj_special_start_id = subj_special_start_id.unsqueeze(1)
        # subj_output = torch.bmm(subj_special_start_id, com_features)
        # obj_special_start_id = obj_special_start_id.unsqueeze(1)
        # obj_output = torch.bmm(obj_special_start_id, com_features)

        pos = (input_ids == self.config.mask_token_id).nonzero(as_tuple=True)

        logits = self.out_proj(
            self.dropout(self.dense(com_features)))

        logits = logits[pos[0], pos[1], self.config.entity_id_st:self.config.entity_id_ed]

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)


class ConditionedParameter(torch.nn.Module):
    def __init__(self, parameter, condition_dim=1024, hidden_dim=128, max_scale=1):
        super().__init__()
        self.parameter_shape = parameter.shape

        if len(self.parameter_shape) == 2:  # condition_dim是从lstm中得到的tensor，然后用linear学习返回到768作为更新的parm_dict
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(
                        hidden_dim, 2 * (parameter.shape[0] + parameter.shape[1]) + 1
                    )
                ),
            )
        elif len(self.parameter_shape) == 1:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(hidden_dim, 2 * parameter.shape[0] + 1)
                ),
            )
        else:
            raise RuntimeError()

        self.max_scale = max_scale

    def forward(self, inputs, grad):

        if len(self.parameter_shape) == 2:
            (
                conditioner_cola,
                conditioner_rowa,
                conditioner_colb,
                conditioner_rowb,
                conditioner_norm,
            ) = self.conditioners(inputs).split(
                [
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    1,
                ],
                dim=-1,
            )

            a = conditioner_rowa.softmax(-1).T @ conditioner_cola
            b = conditioner_rowb.softmax(-1).T @ conditioner_colb

        elif len(self.parameter_shape) == 1:
            a, b, conditioner_norm = self.conditioners(inputs).split(
                [self.parameter_shape[0], self.parameter_shape[0], 1], dim=-1
            )
        else:
            raise RuntimeError()

        return (
                self.max_scale
                * torch.mean(conditioner_norm.sigmoid(), dim=0).squeeze()  # 多条我们直接取mean
                * (grad * a.squeeze() + b.squeeze())
        )


class LSTMConditioner(torch.nn.Module):
    def __init__(
            self,
            vocab_dim=30522,
            embedding_dim=768,
            hidden_dim=256,
            output_dim=1024,
            embedding_init=None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_dim,
            embedding_dim=embedding_dim,
            padding_idx=0,
            _weight=embedding_init,
        )
        self.lstm = PytorchSeq2VecWrapper(
            torch.nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        )
        self.linear = FeedForward(
            input_dim=hidden_dim * 2,
            num_layers=1,
            hidden_dims=[output_dim],
            activations=[torch.nn.Tanh()],
        )

    def forward(self, inputs, masks):
        return self.linear(self.lstm(self.embedding(inputs), masks))  # 1, 64


class OneShotLearner(torch.nn.Module):
    def __init__(
            self,
            model,
            vocab_dim=30522,
            embedding_dim=768,
            hidden_dim=128,
            condition_dim=1024,
            include_set={},
            max_scale=1e-3,
            embedding_init=None,
    ):
        super().__init__()

        self.param2conditioner_map = {
            n: "{}_conditioner".format(n).replace(".", "_")
            for n, p in model.named_parameters()
            if n in include_set
        }

        self.conditioners = torch.nn.ModuleDict(
            {
                self.param2conditioner_map[n]: ConditionedParameter(
                    p,
                    condition_dim,
                    hidden_dim,
                    max_scale=max_scale,
                )
                for n, p in model.named_parameters()
                if n in include_set
            }
        )

        self.condition = LSTMConditioner(
            vocab_dim,
            embedding_dim,
            hidden_dim,
            condition_dim,
            embedding_init=embedding_init,
        )

    def forward(self, inputs, masks, grads=None):
        condition = self.condition(inputs, masks)  # LSTM输出condition
        return {
            p: self.conditioners[self.param2conditioner_map[p]](
                condition,
                grad=grads[p] if grads else None,
            )
            for p, c in self.param2conditioner_map.items()
        }


class BaseKEModule(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.edit_num,
            collate_fn=self.sampler,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            drop_last=True
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.edit_num,
            collate_fn=self.sampler,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            drop_last=True
        )

    def get_logits_orig_params_dict(self, batch):

        with torch.enable_grad():
            logits = self.ex_model.eval()(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

            logits_orig, _ = logits.split([
                len(batch["input_ids"]) - self.edit_num,
                self.edit_num,
            ])
            input_ids = batch['input_ids']
            batch_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[batch_idx, mask_idx, self.entity_id_st:self.entity_id_ed]

            #  get info of gradient
            grads = torch.autograd.grad(
                # cross_entropy
                torch.nn.functional.cross_entropy(
                    mask_logits[-self.edit_num:, :],
                    batch["label"][-self.edit_num:],
                    reduction="none",
                ).mean(-1),
                self.ex_model.parameters(),
            )

        grads = {
            name: grad
            for (name, _), grad in zip(self.ex_model.named_parameters(), grads)
        }

        params_dict = self.ex_learner(
            batch["cond_input_ids"][-self.edit_num:],
            batch["cond_attention_mask"][-self.edit_num:],
            grads=grads,
        )

        return logits_orig.detach()[:, :, self.entity_id_st:self.entity_id_ed], params_dict

    def forward(self, batch, logits_orig=None, params_dict=None):

        if not params_dict:
            logits_orig, params_dict = self.get_logits_orig_params_dict(batch)

        fmodel = make_functional(self.ex_model).eval()

        logits = fmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # add delta theta
            params=[
                params_dict.get(n, 0) + p
                for n, p in self.ex_model.named_parameters()
            ],
        ).logits

        return logits_orig, logits[:, :, self.entity_id_st:self.entity_id_ed], params_dict

    def get_kl_lp_cr(self, logits_orig, logits, label, params_dict, input_ids):
        # Reliability
        pos = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        #
        kl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(
                torch.nn.functional.softmax(logits_orig[pos[0][:-self.edit_num], pos[1][:-self.edit_num], :])),
            torch.distributions.Categorical(
                torch.nn.functional.softmax(logits[pos[0][:-self.edit_num], pos[1][:-self.edit_num], :])),
        )

        # don't update too much params
        lp = sum((p.abs() ** self.hparams.p).mean() ** (1 / self.hparams.p)
                 for p in params_dict.values()) / len(params_dict)

        # ensure the result which has been edited
        cr = torch.nn.functional.cross_entropy(
            logits[pos[0][-self.edit_num:], pos[1][-self.edit_num:], :],
            label[-self.edit_num:],
            reduction="none",
        ).mean(-1)

        return kl, lp, cr

    def training_step(self, batch, batch_idx=None):

        logits_orig, logits, params_dict = self.forward(batch)

        kl, lp, cr = self.get_kl_lp_cr(logits_orig, logits, batch['label'],
                                       params_dict, batch["input_ids"])
        kl = kl.mean(-1)

        # Clip alphas to prevent unbounded regularization growth
        alpha_kl = torch.clamp(self.alpha_kl, 0, 10)
        alpha_lp = torch.clamp(self.alpha_lp, 0, 10)

        # Use relu so that kl < margin_kl produces zero loss (not negative),
        # preventing the model from receiving nonsensical gradients when
        # locality is already well-preserved.
        loss_kl = alpha_kl * torch.relu(kl - self.margin_kl)
        loss_lp = alpha_lp * torch.relu(lp - self.margin_lp)

        if self.hparams.divergences == "both":
            loss = cr + loss_kl + loss_lp
        elif self.hparams.divergences == "kl":
            loss = cr + loss_kl
        elif self.hparams.divergences == "lp":
            loss = cr + loss_lp

        self.log("alpha_kl",
                 self.alpha_kl,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)
        self.log("alpha_lp",
                 self.alpha_lp,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)
        self.log("kl", kl, on_step=True, on_epoch=False, prog_bar=True)
        self.log("lp", lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("cr", cr, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def _eval(self, batch, batch_idx, ):
        logits_orig, params_dict = self.get_logits_orig_params_dict(batch)

        # Compute pre-edit ranks BEFORE make_functional (which patches the model in-place)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch.pop('label')

        # Extract rel_names for per-category analysis (will be popped by cleanup below)
        rel_names = batch.pop('rel_names', None)
        _ = batch.pop('rel_ids', None)  # not needed, rel_names used for category grouping

        with torch.no_grad():
            pre_logits = self.ex_model.eval()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, :, self.entity_id_st:self.entity_id_ed]

        fmodel = make_functional(self.ex_model).eval()
        my_keys = list(batch.keys())
        for k in my_keys:
            if k not in ["input_ids", "attention_mask", "token_type_ids"]:
                batch.pop(k)

        logits = fmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # add delta theta
            params=[
                params_dict.get(n, 0) + p
                for n, p in self.ex_model.named_parameters()
            ],
        ).logits[:, :, self.entity_id_st:self.entity_id_ed]

        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        logits = logits[torch.arange(bsz), mask_idx]

        # Pre-edit ranks
        pre_logits = pre_logits[torch.arange(bsz), mask_idx]
        _, pre_outputs = torch.sort(pre_logits, dim=1, descending=True)
        _, pre_sort = torch.sort(pre_outputs, dim=1)
        pre_ranks = pre_sort[torch.arange(bsz), label].detach().cpu() + 1

        _, outputs = torch.sort(logits, dim=1, descending=True)  # outputs代表
        edit_entity_order = outputs[-self.edit_num:, :]
        edit_input_ids = batch["input_ids"][-self.edit_num:, :]
        edit_labels = label[-self.edit_num:]
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        edit_ranks = ranks[-self.edit_num:]

        # Split rel_names for per-category analysis
        if rel_names is not None:
            loc_rel_names = list(rel_names[:-self.edit_num])
            edit_rel_names = list(rel_names[-self.edit_num:])
        else:
            loc_rel_names = []
            edit_rel_names = []

        return dict(ranks=np.array(ranks), pre_ranks=np.array(pre_ranks),
                    edit_entity_order=np.array(edit_entity_order.detach().cpu()),
                    edit_input_ids=edit_input_ids, edit_labels=edit_labels, edit_ranks=edit_ranks,
                    loc_rel_names=loc_rel_names, edit_rel_names=edit_rel_names)

    def validation_step(self, batch, batch_idx=None):

        result = self._eval(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs):
        if self.epoch % 5 == 0:
            output_dir = f"models/{self.hparams.kge_model_type}/output/{str(self.hparams.data_type)}/{str(self.task_name)}/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(self.ex_learner, os.path.join(output_dir, f"{str(self.epoch)}_params.pt"))

        self.epoch = self.epoch + 1

        loc_ranks = np.concatenate([_['ranks'][:-self.edit_num] for _ in outputs])
        edit_ranks = np.concatenate([_['ranks'][-self.edit_num:] for _ in outputs])
        pre_loc_ranks = np.concatenate([_['pre_ranks'][:-self.edit_num] for _ in outputs])
        pre_edit_ranks = np.concatenate([_['pre_ranks'][-self.edit_num:] for _ in outputs])

        loc_hits5 = (loc_ranks <= 5).mean()
        loc_hits3 = (loc_ranks <= 3).mean()
        loc_hits1 = (loc_ranks <= 1).mean()
        edit_hits5 = (edit_ranks <= 5).mean()
        edit_hits3 = (edit_ranks <= 3).mean()
        edit_hits1 = (edit_ranks <= 1).mean()

        R_edit = edit_ranks.mean()
        Rs_edit = loc_ranks.mean()

        print("Eval/ranks", edit_ranks)
        print("Eval/hits5", edit_hits5)
        print("Eval/hits3", edit_hits3)
        print("Eval/hits1", edit_hits1)
        print("Eval/mean_rank", R_edit)
        print("Eval/mrr", (1. / edit_ranks).mean())
        print("Eval/loc_hits1", loc_hits1)
        print("Eval/loc_hits3", loc_hits3)
        print("Eval/loc_hits5", loc_hits5)
        print("Eval/loc_mean_rank", Rs_edit)
        print("Eval/loc_mrr", (1. / loc_ranks).mean())

        self.log("Eval/loc_hits1", loc_hits1)
        self.log("Eval/loc_hits3", loc_hits3)
        self.log("Eval/loc_hits5", loc_hits5)
        self.log("Eval/loc_mean_rank", Rs_edit)
        self.log("Eval/loc_mrr", (1. / loc_ranks).mean())
        self.log("Eval/hits5", edit_hits5)
        self.log("Eval/hits3", edit_hits3)
        self.log("Eval/hits1", edit_hits1)
        self.log("Eval/mean_rank", R_edit)
        self.log("Eval/mrr", (1. / edit_ranks).mean())

        # --- Per-Category Editing Accuracy (supports FB15k237 & WN18RR) ---
        if 'edit_rel_names' in outputs[0] and len(outputs[0]['edit_rel_names']) > 0:
            try:
                from src.data.relation_categories import get_category, get_target_categories

                model_type = getattr(self.hparams, 'kge_model_type', 'unknown')
                data_type = getattr(self.hparams, 'data_type', 'unknown')
                target_cats = get_target_categories(data_type)

                # Collect all edit rel_names and ranks across batches
                all_edit_rel_names = []
                all_edit_ranks = []
                for out in outputs:
                    all_edit_rel_names.extend(out['edit_rel_names'])
                    all_edit_ranks.extend(out['edit_ranks'])

                # Group by category
                category_ranks = defaultdict(list)
                for rel_name, rank in zip(all_edit_rel_names, all_edit_ranks):
                    cat = get_category(rel_name, data_type)
                    category_ranks[cat].append(rank)

                # Merge low-frequency categories (< 10 samples) into 'other'
                MIN_CATEGORY_COUNT = 10
                merged_ranks = defaultdict(list)
                for cat, ranks_list in category_ranks.items():
                    if len(ranks_list) < MIN_CATEGORY_COUNT:
                        merged_ranks['other'].extend(ranks_list)
                    else:
                        merged_ranks[cat].extend(ranks_list)

                # Compute per-category metrics
                per_category = {}
                for cat, ranks_list in sorted(merged_ranks.items()):
                    ranks_arr = np.array(ranks_list)
                    per_category[cat] = {
                        'count': int(len(ranks_arr)),
                        'edit_hits1': float((ranks_arr <= 1).mean()),
                        'edit_hits3': float((ranks_arr <= 3).mean()),
                        'edit_hits5': float((ranks_arr <= 5).mean()),
                        'edit_mean_rank': float(ranks_arr.mean()),
                    }

                # Save per-category JSON
                output_dir = f"results/{model_type}/{data_type}/"
                os.makedirs(output_dir, exist_ok=True)
                json_path = os.path.join(output_dir, f"per_category_epoch_{self.epoch}.json")
                with open(json_path, 'w') as f:
                    json.dump(per_category, f, indent=2, ensure_ascii=False)

                # Save comprehensive text result (overall + per-category)
                result_dir = f"result/{model_type}/{data_type}/"
                os.makedirs(result_dir, exist_ok=True)
                result_path = os.path.join(result_dir, f"epoch_{self.epoch}.txt")
                with open(result_path, 'w') as f:
                    f.write(f"========== {model_type} Epoch {self.epoch} ==========\n\n")
                    f.write("【Overall Results】\n")
                    f.write(f"  count: {len(edit_ranks)}\n")
                    f.write(f"  edit_hits1: {edit_hits1:.10f}\n")
                    f.write(f"  edit_hits3: {edit_hits3:.10f}\n")
                    f.write(f"  edit_hits5: {edit_hits5:.10f}\n")
                    f.write(f"  edit_mean_rank: {R_edit:.10f}\n")
                    f.write(f"  edit_mrr: {(1. / edit_ranks).mean():.10f}\n")
                    f.write(f"  loc_hits1: {loc_hits1:.10f}\n")
                    f.write(f"  loc_hits3: {loc_hits3:.10f}\n")
                    f.write(f"  loc_hits5: {loc_hits5:.10f}\n")
                    f.write(f"  loc_mean_rank: {Rs_edit:.10f}\n")
                    f.write(f"  loc_mrr: {(1. / loc_ranks).mean():.10f}\n\n")
                    f.write("【Per-Category Editing Accuracy】\n")
                    for cat in sorted(per_category.keys()):
                        info = per_category[cat]
                        marker = ' ★' if cat in target_cats else ''
                        f.write(f"  {cat:15s} | count={info['count']:4d} | hits@1={info['edit_hits1']:.6f} | hits@3={info['edit_hits3']:.6f} | hits@5={info['edit_hits5']:.6f} | mean_rank={info['edit_mean_rank']:.2f}{marker}\n")
                    f.write("\n")

                print(f"\n【Per-Category Results】 {model_type} epoch {self.epoch}")
                for cat in sorted(per_category.keys()):
                    info = per_category[cat]
                    marker = ' ★' if cat in target_cats else ''
                    print(f"  {cat:15s} | count={info['count']:4d} | hits@1={info['edit_hits1']:.6f} | hits@3={info['edit_hits3']:.6f} | hits@5={info['edit_hits5']:.6f} | mean_rank={info['edit_mean_rank']:.2f}{marker}")
                print()
            except Exception as e:
                print(f"[Per-Category] Skipped: {e}")

        return super().validation_epoch_end(outputs)

    def sample(
            self,
            sentences,
            condition,
            logits_orig=None,
            params_dict=None,
            stop_condition=None,
    ):
        len_sent = len(sentences)
        with torch.no_grad():
            logits_orig, logits, params_dict = self.forward(
                {
                    k: v.to(self.device)
                    for k, v in self.val_dataset.get_batch(
                    sentences, condition).items()
                },
                logits_orig=logits_orig,
                params_dict=params_dict,
            )

            n_iter = 1
            if stop_condition is not None and stop_condition(
                    condition, logits, n_iter):
                model_tmp = deepcopy(self.model)
                params_dict_tmp = deepcopy(params_dict)

                while stop_condition(condition, logits, n_iter):
                    for n, p in self.model.named_parameters():
                        p.data += params_dict.get(n, 0)

                    _, logits, params_dict = self.forward({
                        k: v.to(self.device)
                        for k, v in self.val_dataset.get_batch(
                            sentences, condition).items()
                    })
                    params_dict_tmp = {
                        k: v + params_dict[k]
                        for k, v in params_dict_tmp.items()
                    }
                    n_iter += 1

                self.model = model_tmp
                params_dict = params_dict_tmp

            return logits_orig, logits[:len_sent], params_dict

    def on_before_zero_grad(self, optimizer):
        self.alpha_kl.data = torch.where(
            self.alpha_kl.data < 0,
            torch.full_like(self.alpha_kl.data, 0),
            self.alpha_kl.data,
        )
        self.alpha_lp.data = torch.where(
            self.alpha_lp.data < 0,
            torch.full_like(self.alpha_lp.data, 0),
            self.alpha_lp.data,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            [
                {
                    "params": self.ex_learner.parameters(),
                    "lr": self.hparams.lr,
                },
                {
                    "params": [self.alpha_kl, self.alpha_lp],
                    "lr": self.hparams.lr_alpha,
                },
            ],
            centered=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }]


class KGEditor(BaseKEModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(
            self.hparams.ex_model_checkpoint,
        )

        config.ex_size = self.hparams.ex_size
        config.kb_layer = [int(i) for i in self.hparams.kb_layer.split(',') if i != '']

        self.ex_model = EXBertForMaskedLM.from_pretrained(
            self.hparams.ex_model_checkpoint, config=config).eval()

        data = dataset_mapping[self.hparams.kge_model_type](Args(**kwargs))
        data.setup()

        self.__dict__.update(data.get_config())

        self.task_name = self.hparams.task_name

        self.epoch = 0
        self.edit_num = self.hparams.edit_num

        self.ex_learner = OneShotLearner(
            self.ex_model,
            vocab_dim=self.ex_model.bert.embeddings.word_embeddings.weight.data.
            shape[0],
            embedding_dim=self.ex_model.bert.embeddings.word_embeddings.weight.
            data.shape[1],
            hidden_dim=128,
            condition_dim=1024,
            include_set={
                n
                for n, _ in self.ex_model.named_parameters()
                if ("dense_in_ex" in n.lower() or "dense_out_ex" in n.lower()) and "bias" not in n.lower()
            },
            max_scale=self.hparams.max_scale,
            embedding_init=self.ex_model.bert.embeddings.word_embeddings.weight.
            data,
        )

        self.alpha_kl = torch.nn.Parameter(torch.ones(()))
        self.alpha_kl.register_hook(lambda grad: -grad)

        self.alpha_lp = torch.nn.Parameter(torch.ones(()))
        self.alpha_lp.register_hook(lambda grad: -grad)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_flipped = pl.metrics.Accuracy()

        self.register_buffer("margin_kl",
                             torch.tensor(self.hparams.margin_kl_max))
        self.register_buffer("margin_lp",
                             torch.tensor(self.hparams.margin_lp_max))
        self.running_flipped = []


class DualEditor(BaseKEModule):
    """Knowledge editor with Split FFN (general + memory sub-spaces).

    The intermediate FFN dimension is split into a frozen general sub-space
    (language capacity) and a trainable memory sub-space (factual knowledge).
    During editing, only the memory pathway is updated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(
            self.hparams.ex_model_checkpoint,
        )

        config.ex_size = self.hparams.ex_size
        config.memory_ratio = self.hparams.memory_ratio
        config.dual_layer = [int(i) for i in self.hparams.dual_layer.split(',') if i != '']

        self.ex_model = DualBertForMaskedLM.from_pretrained(
            self.hparams.ex_model_checkpoint, config=config).eval()

        data = dataset_mapping[self.hparams.kge_model_type](Args(**kwargs))
        data.setup()

        self.__dict__.update(data.get_config())

        self.task_name = self.hparams.task_name

        self.epoch = 0
        self.edit_num = self.hparams.edit_num

        # OneShotLearner targets only memory-path parameters
        self.ex_learner = OneShotLearner(
            self.ex_model,
            vocab_dim=self.ex_model.bert.embeddings.word_embeddings.weight.data.
            shape[0],
            embedding_dim=self.ex_model.bert.embeddings.word_embeddings.weight.
            data.shape[1],
            hidden_dim=128,
            condition_dim=1024,
            include_set={
                n
                for n, _ in self.ex_model.named_parameters()
                if ("dense_memory" in n.lower()) and "bias" not in n.lower()
            },
            max_scale=self.hparams.max_scale,
            embedding_init=self.ex_model.bert.embeddings.word_embeddings.weight.
            data,
        )

        self.alpha_kl = torch.nn.Parameter(torch.ones(()))
        self.alpha_kl.register_hook(lambda grad: -grad)

        self.alpha_lp = torch.nn.Parameter(torch.ones(()))
        self.alpha_lp.register_hook(lambda grad: -grad)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_flipped = pl.metrics.Accuracy()

        self.register_buffer("margin_kl",
                             torch.tensor(self.hparams.margin_kl_max))
        self.register_buffer("margin_lp",
                             torch.tensor(self.hparams.margin_lp_max))
        self.running_flipped = []

        # === SI路径积分指导的神经元重要性重排 ===
        _saved_memory_pos = getattr(getattr(self, 'sampler', None), 'memory_pos', None)
        try:
            sample_size = min(64, len(self.data_train))
            if sample_size > 0:
                sample_features = [self.data_train[i] for i in range(sample_size)]
                batch = self.sampler(sample_features)
                device = next(self.ex_model.parameters()).device
                importances = self.ex_model.compute_neuron_importance(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                )
                if importances:
                    self.ex_model.reorganize_by_importance(importances)
                    print(f"[DualEditor] SI importance reorg OK for layers {list(importances.keys())}")
        except Exception as e:
            print(f"[DualEditor] SI reorg skipped: {e}")
        finally:
            if _saved_memory_pos is not None:
                self.sampler.memory_pos = _saved_memory_pos


class SIEditor(BaseKEModule):
    """Knowledge editor with SI-guided dynamic neuron selection.

    Uses a standard EXBertForMaskedLM (no fixed FFN split). SI computes
    per-neuron importance on a sample of training data, then scales the
    OneShotLearner parameter deltas: important neurons get small updates
    (protect language capacity), unimportant neurons get larger updates
    (edit factual knowledge). Controlled by --si_mask_strength.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(
            self.hparams.ex_model_checkpoint,
        )
        config.ex_size = self.hparams.ex_size
        config.kb_layer = [int(i) for i in self.hparams.kb_layer.split(',') if i != '']

        self.ex_model = EXBertForMaskedLM.from_pretrained(
            self.hparams.ex_model_checkpoint, config=config).eval()

        data = dataset_mapping[self.hparams.kge_model_type](Args(**kwargs))
        data.setup()
        self.__dict__.update(data.get_config())

        self.task_name = self.hparams.task_name
        self.epoch = 0
        self.edit_num = self.hparams.edit_num

        # OneShotLearner — targets all FFN weights (no dual split)
        self.ex_learner = OneShotLearner(
            self.ex_model,
            vocab_dim=self.ex_model.bert.embeddings.word_embeddings.weight.data.
            shape[0],
            embedding_dim=self.ex_model.bert.embeddings.word_embeddings.weight.
            data.shape[1],
            hidden_dim=self.hparams.hidden_dim,
            condition_dim=1024,
            include_set={
                n
                for n, _ in self.ex_model.named_parameters()
                if ("intermediate" in n.lower() or ("dense_in_ex" in n.lower() and not self.hparams.ablate_ex_params))
                and "bias" not in n.lower()
            },
            max_scale=self.hparams.max_scale,
            embedding_init=self.ex_model.bert.embeddings.word_embeddings.weight.
            data,
        )

        self.alpha_kl = torch.nn.Parameter(torch.ones(()))
        self.alpha_kl.register_hook(lambda grad: -grad)

        self.alpha_lp = torch.nn.Parameter(torch.ones(()))
        self.alpha_lp.register_hook(lambda grad: -grad)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_flipped = pl.metrics.Accuracy()

        self.register_buffer("margin_kl",
                             torch.tensor(self.hparams.margin_kl_max))
        self.register_buffer("margin_lp",
                             torch.tensor(self.hparams.margin_lp_max))
        self.running_flipped = []

        # SI mask strength (0 = no masking, 1 = fully freeze important neurons)
        self.si_mask_strength = getattr(self.hparams, 'si_mask_strength', 0.9)

        # Compute neuron importance once at init
        self._neuron_importance = self._compute_neuron_importance()

    # ── SI importance computation ──────────────────────────────────────────

    def _compute_neuron_importance(self):
        """Compute per-neuron importance for all intermediate layers via MLM proxy.

        Returns {layer_idx: importance_tensor} where importance ∈ [0, 1].
        """
        self.ex_model.eval()

        if getattr(self.hparams, 'random_si', False):
            inter_layers = []
            for i, layer in enumerate(self.ex_model.bert.encoder.layer):
                if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
                    inter_layers.append(i)
            if not inter_layers:
                return {}
            device = next(self.ex_model.parameters()).device
            importances = {
                idx: torch.rand(self.ex_model.bert.encoder.layer[idx].intermediate.dense.out_features, device=device)
                for idx in inter_layers
            }
            for idx in inter_layers:
                imp = importances[idx]
                imp = imp / (imp.max() + 1e-8)
                importances[idx] = imp
            print(f"[SIEditor] Random SI importance for layers {inter_layers}")
            return importances

        inter_layers = []
        for i, layer in enumerate(self.ex_model.bert.encoder.layer):
            if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
                inter_layers.append(i)
        if not inter_layers:
            return {}

        sample_size = min(64, len(self.data_train))
        if sample_size == 0:
            return {}
        sample_features = [self.data_train[j] for j in range(sample_size)]
        batch = self.sampler(sample_features)
        device = next(self.ex_model.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Register forward/backward hooks on intermediate layers
        inter_outputs = {}
        inter_grads = {}
        hooks = []
        for idx in inter_layers:
            layer = self.ex_model.bert.encoder.layer[idx]
            hooks.append(layer.intermediate.register_forward_hook(
                lambda mod, inp, out, i=idx: inter_outputs.__setitem__(i, out)))
            hooks.append(layer.intermediate.register_full_backward_hook(
                lambda mod, grad_inp, grad_out, i=idx: inter_grads.__setitem__(i, grad_out[0])))

        # MLM proxy: randomly mask 15% of non-special tokens
        masked_input_ids = input_ids.clone()
        excluded = (
            (input_ids == getattr(self.ex_model.config, 'pad_token_id', 0))
            | (input_ids == 101) | (input_ids == 102) | (input_ids == 103))
        rand = torch.rand(input_ids.shape, device=device)
        to_mask = (rand < 0.15) & ~excluded
        mlm_labels = input_ids.clone()
        mlm_labels[~to_mask] = -100
        masked_input_ids[to_mask] = 103

        if to_mask.sum() < 2:
            for h in hooks:
                h.remove()
            self.ex_model.zero_grad()
            return {}

        outputs = self.ex_model(
            input_ids=masked_input_ids, attention_mask=attention_mask, labels=None)
        loss = nn.CrossEntropyLoss()(outputs.logits[to_mask], mlm_labels[to_mask])
        loss.backward()

        # |grad * output| → importance
        importances = {}
        for idx in inter_layers:
            h = inter_outputs[idx].detach()
            g = inter_grads[idx].detach()
            imp = (g * h).abs().mean(dim=[0, 1])           # (3072,)
            imp = imp + 0.1 * g.abs().mean(dim=[0, 1])     # smooth
            imp = imp / (imp.max() + 1e-8)                  # normalize → [0, 1]
            importances[idx] = imp

        for h in hooks:
            h.remove()
        self.ex_model.zero_grad()
        print(f"[SIEditor] SI importance computed for layers {list(importances.keys())}")
        return importances

    # ── SI masking applied in forward ──────────────────────────────────────

    def forward(self, batch, logits_orig=None, params_dict=None):
        if not params_dict:
            logits_orig, params_dict = self.get_logits_orig_params_dict(batch)

        # Apply SI mask: important ← small delta, unimportant ← large delta
        if self.si_mask_strength > 0 and self._neuron_importance:
            for n, delta in list(params_dict.items()):
                if 'encoder.layer.' not in n:
                    continue
                parts = n.split('.')
                try:
                    idx_pos = parts.index('layer') + 1
                    layer_idx = int(parts[idx_pos])
                except (ValueError, IndexError):
                    continue
                if layer_idx not in self._neuron_importance:
                    continue

                imp = self._neuron_importance[layer_idx].to(delta.device)  # (3072,)
                # Normalise if this isn't an intermediate-sized param
                if 'intermediate.dense.weight' in n:
                    # (3072, 768) — each row is a neuron
                    mask = 1.0 - self.si_mask_strength * imp
                    params_dict[n] = delta * mask.unsqueeze(1)
                elif 'dense_in_ex.weight' in n:
                    # nn.Linear(3072, ex_size) → weight (ex_size, 3072), each column is a neuron
                    mask = 1.0 - self.si_mask_strength * imp
                    params_dict[n] = delta * mask.unsqueeze(0)

        fmodel = make_functional(self.ex_model).eval()
        logits = fmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=[
                params_dict.get(n, 0) + p
                for n, p in self.ex_model.named_parameters()
            ],
        ).logits

        return logits_orig, logits[:, :, self.entity_id_st:self.entity_id_ed], params_dict


class KnowledgeEditor(BaseKEModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForMaskedLM.from_pretrained(
            self.hparams.model_checkpoint).eval()

        self.task_name = self.hparams.task_name

        data = dataset_mapping[self.hparams.kge_model_type](Args(**kwargs))
        data.setup()
        self.__dict__.update(data.get_config())

        self.epoch = 0
        self.edit_num = self.hparams.edit_num

        self.learner = OneShotLearner(
            self.model,
            vocab_dim=self.model.bert.embeddings.word_embeddings.weight.data.
            shape[0],
            embedding_dim=self.model.bert.embeddings.word_embeddings.weight.
            data.shape[1],
            hidden_dim=128,
            condition_dim=1024,
            include_set={
                n
                for n, _ in self.model.named_parameters()
                if all(e not in n.lower() for e in (
                    "bias",
                    "norm",
                    "embeddings",
                    "cls",
                    "pooler",
                    "shared",
                    "embed",
                    "positions",
                ))
            },
            max_scale=self.hparams.max_scale,
            embedding_init=self.model.bert.embeddings.word_embeddings.weight.
            data,
        )

        self.alpha_kl = torch.nn.Parameter(torch.ones(()))
        self.alpha_kl.register_hook(lambda grad: -grad)

        self.alpha_lp = torch.nn.Parameter(torch.ones(()))
        self.alpha_lp.register_hook(lambda grad: -grad)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_flipped = pl.metrics.Accuracy()

        self.register_buffer("margin_kl",
                             torch.tensor(self.hparams.margin_kl_max))
        self.register_buffer("margin_lp",
                             torch.tensor(self.hparams.margin_lp_max))
        self.running_flipped = []

    def get_logits_orig_params_dict(self, batch):

        with torch.enable_grad():
            logits = self.model.eval()(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

            logits_orig, _ = logits.split([
                len(batch["input_ids"]) - self.edit_num,
                self.edit_num,
            ])
            input_ids = batch['input_ids']
            batch_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[batch_idx, mask_idx, self.entity_id_st:self.entity_id_ed]

            #  get info of gradient
            grads = torch.autograd.grad(
                # cross_entropy
                torch.nn.functional.cross_entropy(
                    mask_logits[-self.edit_num:, :],
                    batch["label"][-self.edit_num:],
                    reduction="none",
                ).mean(-1),
                self.model.parameters(),
            )

        grads = {
            name: grad
            for (name, _), grad in zip(self.model.named_parameters(), grads)
        }

        params_dict = self.learner(
            batch["cond_input_ids"][-self.edit_num:],
            batch["cond_attention_mask"][-self.edit_num:],
            grads=grads,
        )

        return logits_orig.detach()[:, :, self.entity_id_st:self.entity_id_ed], params_dict

    def forward(self, batch, logits_orig=None, params_dict=None):

        if not params_dict:
            logits_orig, params_dict = self.get_logits_orig_params_dict(batch)

        fmodel = make_functional(self.model).eval()

        logits = fmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # add delta theta
            params=[
                params_dict.get(n, 0) + p
                for n, p in self.model.named_parameters()
            ],
        ).logits

        return logits_orig, logits[:, :, self.entity_id_st:self.entity_id_ed], params_dict

    def _eval(self, batch, batch_idx, ):
        logits_orig, params_dict = self.get_logits_orig_params_dict(batch)

        # Compute pre-edit ranks BEFORE make_functional (which patches the model in-place)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch.pop('label')

        # Extract rel_names for per-category analysis (will be popped by cleanup below)
        rel_names = batch.pop('rel_names', None)
        _ = batch.pop('rel_ids', None)  # not needed, rel_names used for category grouping

        with torch.no_grad():
            pre_logits = self.model.eval()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, :, self.entity_id_st:self.entity_id_ed]

        fmodel = make_functional(self.model).eval()
        my_keys = list(batch.keys())
        for k in my_keys:
            if k not in ["input_ids", "attention_mask", "token_type_ids"]:
                batch.pop(k)

        logits = fmodel(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # add delta theta
            params=[
                params_dict.get(n, 0) + p
                for n, p in self.model.named_parameters()
            ],
        ).logits[:, :, self.entity_id_st:self.entity_id_ed]

        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        logits = logits[torch.arange(bsz), mask_idx]

        # Pre-edit ranks
        pre_logits = pre_logits[torch.arange(bsz), mask_idx]
        _, pre_outputs = torch.sort(pre_logits, dim=1, descending=True)
        _, pre_sort = torch.sort(pre_outputs, dim=1)
        pre_ranks = pre_sort[torch.arange(bsz), label].detach().cpu() + 1

        _, outputs = torch.sort(logits, dim=1, descending=True)  # outputs代表
        edit_entity_order = outputs[-self.edit_num:, :]
        edit_input_ids = batch["input_ids"][-self.edit_num:, :]
        edit_labels = label[-self.edit_num:]
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        edit_ranks = ranks[-self.edit_num:]

        # Split rel_names for per-category analysis
        if rel_names is not None:
            loc_rel_names = list(rel_names[:-self.edit_num])
            edit_rel_names = list(rel_names[-self.edit_num:])
        else:
            loc_rel_names = []
            edit_rel_names = []

        return dict(ranks=np.array(ranks), pre_ranks=np.array(pre_ranks),
                    edit_entity_order=np.array(edit_entity_order.detach().cpu()),
                    edit_input_ids=edit_input_ids, edit_labels=edit_labels, edit_ranks=edit_ranks,
                    loc_rel_names=loc_rel_names, edit_rel_names=edit_rel_names)

    def validation_step(self, batch, batch_idx=None):

        result = self._eval(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs):
        if self.epoch % 5 == 0:
            output_dir = f"models/{self.hparams.kge_model_type}/output/{str(self.hparams.data_type)}/{str(self.task_name)}/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(self.learner, os.path.join(output_dir, f"{str(self.epoch)}_params.pt"))

        self.epoch = self.epoch + 1

        loc_ranks = np.concatenate([_['ranks'][:-self.edit_num] for _ in outputs])
        edit_ranks = np.concatenate([_['ranks'][-self.edit_num:] for _ in outputs])
        pre_loc_ranks = np.concatenate([_['pre_ranks'][:-self.edit_num] for _ in outputs])
        pre_edit_ranks = np.concatenate([_['pre_ranks'][-self.edit_num:] for _ in outputs])

        loc_hits5 = (loc_ranks <= 5).mean()
        loc_hits3 = (loc_ranks <= 3).mean()
        loc_hits1 = (loc_ranks <= 1).mean()
        edit_hits5 = (edit_ranks <= 5).mean()
        edit_hits3 = (edit_ranks <= 3).mean()
        edit_hits1 = (edit_ranks <= 1).mean()

        R_edit = edit_ranks.mean()
        Rs_edit = loc_ranks.mean()

        print("Eval/ranks", edit_ranks)
        print("Eval/hits5", edit_hits5)
        print("Eval/hits3", edit_hits3)
        print("Eval/hits1", edit_hits1)
        print("Eval/mean_rank", R_edit)
        print("Eval/mrr", (1. / edit_ranks).mean())
        print("Eval/loc_hits1", loc_hits1)
        print("Eval/loc_hits3", loc_hits3)
        print("Eval/loc_hits5", loc_hits5)
        print("Eval/loc_mean_rank", Rs_edit)
        print("Eval/loc_mrr", (1. / loc_ranks).mean())

        self.log("Eval/loc_hits1", loc_hits1)
        self.log("Eval/loc_hits3", loc_hits3)
        self.log("Eval/loc_hits5", loc_hits5)
        self.log("Eval/loc_mean_rank", Rs_edit)
        self.log("Eval/loc_mrr", (1. / loc_ranks).mean())
        self.log("Eval/hits5", edit_hits5)
        self.log("Eval/hits3", edit_hits3)
        self.log("Eval/hits1", edit_hits1)
        self.log("Eval/mean_rank", R_edit)
        self.log("Eval/mrr", (1. / edit_ranks).mean())

        # --- Per-Category Editing Accuracy (supports FB15k237 & WN18RR) ---
        if 'edit_rel_names' in outputs[0] and len(outputs[0]['edit_rel_names']) > 0:
            try:
                from src.data.relation_categories import get_category, get_target_categories

                model_type = getattr(self.hparams, 'kge_model_type', 'unknown')
                data_type = getattr(self.hparams, 'data_type', 'unknown')
                target_cats = get_target_categories(data_type)

                # Collect all edit rel_names and ranks across batches
                all_edit_rel_names = []
                all_edit_ranks = []
                for out in outputs:
                    all_edit_rel_names.extend(out['edit_rel_names'])
                    all_edit_ranks.extend(out['edit_ranks'])

                # Group by category
                category_ranks = defaultdict(list)
                for rel_name, rank in zip(all_edit_rel_names, all_edit_ranks):
                    cat = get_category(rel_name, data_type)
                    category_ranks[cat].append(rank)

                # Merge low-frequency categories (< 10 samples) into 'other'
                MIN_CATEGORY_COUNT = 10
                merged_ranks = defaultdict(list)
                for cat, ranks_list in category_ranks.items():
                    if len(ranks_list) < MIN_CATEGORY_COUNT:
                        merged_ranks['other'].extend(ranks_list)
                    else:
                        merged_ranks[cat].extend(ranks_list)

                # Compute per-category metrics
                per_category = {}
                for cat, ranks_list in sorted(merged_ranks.items()):
                    ranks_arr = np.array(ranks_list)
                    per_category[cat] = {
                        'count': int(len(ranks_arr)),
                        'edit_hits1': float((ranks_arr <= 1).mean()),
                        'edit_hits3': float((ranks_arr <= 3).mean()),
                        'edit_hits5': float((ranks_arr <= 5).mean()),
                        'edit_mean_rank': float(ranks_arr.mean()),
                    }

                # Save per-category JSON
                output_dir = f"results/{model_type}/{data_type}/"
                os.makedirs(output_dir, exist_ok=True)
                json_path = os.path.join(output_dir, f"per_category_epoch_{self.epoch}.json")
                with open(json_path, 'w') as f:
                    json.dump(per_category, f, indent=2, ensure_ascii=False)

                # Save comprehensive text result (overall + per-category)
                result_dir = f"result/{model_type}/{data_type}/"
                os.makedirs(result_dir, exist_ok=True)
                result_path = os.path.join(result_dir, f"epoch_{self.epoch}.txt")
                with open(result_path, 'w') as f:
                    f.write(f"========== {model_type} Epoch {self.epoch} ==========\n\n")
                    f.write("【Overall Results】\n")
                    f.write(f"  count: {len(edit_ranks)}\n")
                    f.write(f"  edit_hits1: {edit_hits1:.10f}\n")
                    f.write(f"  edit_hits3: {edit_hits3:.10f}\n")
                    f.write(f"  edit_hits5: {edit_hits5:.10f}\n")
                    f.write(f"  edit_mean_rank: {R_edit:.10f}\n")
                    f.write(f"  edit_mrr: {(1. / edit_ranks).mean():.10f}\n")
                    f.write(f"  loc_hits1: {loc_hits1:.10f}\n")
                    f.write(f"  loc_hits3: {loc_hits3:.10f}\n")
                    f.write(f"  loc_hits5: {loc_hits5:.10f}\n")
                    f.write(f"  loc_mean_rank: {Rs_edit:.10f}\n")
                    f.write(f"  loc_mrr: {(1. / loc_ranks).mean():.10f}\n\n")
                    f.write("【Per-Category Editing Accuracy】\n")
                    for cat in sorted(per_category.keys()):
                        info = per_category[cat]
                        marker = ' ★' if cat in target_cats else ''
                        f.write(f"  {cat:15s} | count={info['count']:4d} | hits@1={info['edit_hits1']:.6f} | hits@3={info['edit_hits3']:.6f} | hits@5={info['edit_hits5']:.6f} | mean_rank={info['edit_mean_rank']:.2f}{marker}\n")
                    f.write("\n")

                print(f"\n【Per-Category Results】 {model_type} epoch {self.epoch}")
                for cat in sorted(per_category.keys()):
                    info = per_category[cat]
                    marker = ' ★' if cat in target_cats else ''
                    print(f"  {cat:15s} | count={info['count']:4d} | hits@1={info['edit_hits1']:.6f} | hits@3={info['edit_hits3']:.6f} | hits@5={info['edit_hits5']:.6f} | mean_rank={info['edit_mean_rank']:.2f}{marker}")
                print()
            except Exception as e:
                print(f"[Per-Category] Skipped: {e}")

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            [
                {
                    "params": self.learner.parameters(),
                    "lr": self.hparams.lr,
                },
                {
                    "params": [self.alpha_kl, self.alpha_lp],
                    "lr": self.hparams.lr_alpha,
                },
            ],
            centered=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }]


model_mapping = {
    'KGEditor': KGEditor,
    'DualEditor': DualEditor,
    'SIEditor': SIEditor,
    'KE': KnowledgeEditor,
}