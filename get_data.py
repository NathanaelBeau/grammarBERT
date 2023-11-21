from dataset.preprocess_data import get_data, preprocess_examples, get_act_dict

# Récupère le dataset filtré
dataset = get_data()

# Récupère les dictionnaires d'actions et les primitives
act_dict, primitives = get_act_dict()

# Prétraitement des exemples avec les arguments nécessaires
preprocess_examples(dataset, act_dict, primitives)
