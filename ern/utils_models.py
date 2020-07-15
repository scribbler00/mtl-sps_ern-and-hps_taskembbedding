import sys

sys.path.append("../../dies/")
# from dies.utils import set_random_states
from dies.multitask_architectures import HPSNetwork, CSNetwork, EmergingRelationNetwork
from dies.embedding import Embedding
import numpy as np
import torch, random
from dies.utils_pytorch import xavier_init_normal
from dies.mlp import MultiLayerPeceptron
from dies.utils import get_structure


def get_hps_model(
    ann_structure,
    device,
    no_input_features,
    no_tasks,
    use_emb=False,
    temporal_embedding=[367, 25],
    additional_embs=[],
    dropout=0.5,
):
    set_random_states(42)
    embedding_dropout = 0.25

    sizes_cat = temporal_embedding + additional_embs

    embedding_module = Embedding(sizes_cat, embedding_dropout=embedding_dropout)
    if not use_emb:
        embedding_module = None

    ann_model = HPSNetwork(
        shared_layer_params=ann_structure,
        separate_layer_params=[ann_structure[-1], ann_structure[-1] // 2, 1],
        number_of_tasks=no_tasks,
        use_batch_norm=True,
        dropout=dropout,
        embedding_module=embedding_module,
    ).to(device)

    ann_model = ann_model.apply(xavier_init_normal)

    return ann_model


def get_cs_model(
    ann_structure,
    device,
    no_input_features,
    no_tasks,
    use_emb=False,
    temporal_embedding=[367, 25],
    additional_embs=[],
    dropout=0.5,
):
    set_random_states(42)
    embedding_dropout = 0.25

    sizes_cat = temporal_embedding + additional_embs

    embedding_module = Embedding(sizes_cat, embedding_dropout=embedding_dropout)
    if not use_emb:
        embedding_module = None

    ann_model = CSNetwork(
        layer_params=[ann_structure for _ in range(no_tasks)],
        number_of_tasks=no_tasks,
        use_batch_norm=True,
        dropout=dropout,
        embedding_module=embedding_module,
    ).to(device)

    ann_model = ann_model.apply(xavier_init_normal)

    return ann_model


def get_sn_model(
    ann_structure,
    device,
    no_input_features,
    no_tasks,
    use_emb=False,
    temporal_embedding=[367, 24],
    additional_embs=[],
    dropout=0.5,
    num_supaces=2,
):

    set_random_states(42)
    embedding_dropout = 0.25

    sizes_cat = temporal_embedding + additional_embs

    embedding_module = Embedding(sizes_cat, embedding_dropout=embedding_dropout)
    if not use_emb:
        embedding_module = None

    ann_model = CSNetwork(
        layer_params=[ann_structure for _ in range(no_tasks)],
        number_of_tasks=no_tasks,
        use_batch_norm=True,
        dropout=dropout,
        embedding_module=embedding_module,
        num_supaces=num_supaces,
        as_sluice_nw=True,
    ).to(device)

    ann_model = ann_model.apply(xavier_init_normal)

    return ann_model


def get_ern_model(
    ann_structure,
    device,
    no_input_features,
    no_tasks,
    use_emb=False,
    temporal_embedding=[367, 24],
    additional_embs=[],
    dropout=0.5,
):

    set_random_states(42)
    embedding_dropout = 0.25

    sizes_cat = temporal_embedding + additional_embs

    embedding_module = Embedding(sizes_cat, embedding_dropout=embedding_dropout)
    if not use_emb:
        embedding_module = None

    ann_model = EmergingRelationNetwork(
        layer_params=[ann_structure for _ in range(no_tasks)],
        number_of_tasks=no_tasks,
        use_batch_norm=True,
        dropout=dropout,
        embedding_module=embedding_module,
    ).to(device)

    ann_model = ann_model.apply(xavier_init_normal)

    return ann_model


def get_mlp_model(
    ann_structure,
    device,
    no_input_features,
    use_emb=False,
    temporal_embedding=[367, 24],
    additional_embs=[],
    dropout=0.5,
):
    set_random_states(42)
    embedding_dropout = 0.25

    sizes_cat = temporal_embedding + additional_embs

    set_random_states(42)

    embedding_module = Embedding(sizes_cat, embedding_dropout=embedding_dropout)
    if not use_emb:
        embedding_module = None

    ann_model = MultiLayerPeceptron(
        input_size=no_input_features,
        ann_structure=ann_structure,  # separate_layer_params = ann_structure,\
        #                         number_of_tasks=no_output_features,\
        use_batch_norm=True,
        dropout=dropout,
        embedding_module=embedding_module,
    ).to(device)

    ann_model = ann_model.apply(xavier_init_normal)

    return ann_model


def set_random_states(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_model(
    model_type,
    no_input_features,
    config,
    device,
    no_tasks,
    use_emb,
    files,
    dropout,
    num_supaces,
):
    if model_type == "hps":

        ann_structure = ann_structure = [no_input_features] + get_structure(
            initial_size=no_input_features * config["size_multiplier"],
            percental_reduce=config["percental_reduce"],
            min_value=11,
            final_outputs=10,
        )

        ann_model = get_hps_model(
            ann_structure,
            device,
            no_input_features,
            no_tasks,
            use_emb=use_emb,
            dropout=dropout,
        )
    elif model_type == "cs":
        ann_structure = [no_input_features // no_tasks] + get_structure(
            initial_size=no_input_features // no_tasks * config["size_multiplier"],
            percental_reduce=config["percental_reduce"],
            min_value=11,
            final_outputs=[5, 1],
        )
        ann_model = get_cs_model(
            ann_structure,
            device,
            no_input_features,
            no_tasks,
            use_emb=use_emb,
            dropout=dropout,
        )
    elif model_type == "sn":
        ann_structure = [no_input_features // no_tasks] + get_structure(
            initial_size=no_input_features // no_tasks * config["size_multiplier"],
            percental_reduce=config["percental_reduce"],
            min_value=11,
            final_outputs=[5, 1],
        )
        ann_model = get_sn_model(
            ann_structure,
            device,
            no_input_features,
            no_tasks,
            use_emb=use_emb,
            dropout=dropout,
            num_supaces=int(num_supaces),
        )
    elif model_type == "ern":
        ann_structure = [no_input_features // no_tasks] + get_structure(
            initial_size=no_input_features // no_tasks * config["size_multiplier"],
            percental_reduce=config["percental_reduce"],
            min_value=11,
            final_outputs=[5, 1],
        )
        ann_model = get_ern_model(
            ann_structure,
            device,
            no_input_features,
            no_tasks,
            use_emb=use_emb,
            dropout=dropout,
        )
    elif model_type == "mlp":
        ann_structure = ann_structure = get_structure(
            no_input_features * config["size_multiplier"],
            config["percental_reduce"],
            min_value=11,
            final_outputs=[5, 1,],
        )
        ann_model = get_mlp_model(
            ann_structure,
            device,
            no_input_features,
            use_emb=use_emb,
            additional_embs=[len(files) + 1],
            dropout=dropout,
        )

    return ann_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
