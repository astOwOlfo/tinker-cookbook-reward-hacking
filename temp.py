import pickle

for filename in [
    "eval_results/evil_genie.pickle.model-tinker___1e1e6607-7cc8-57a8-ae7f-21745560215b_train_0_sampler_weights_000072",
    "eval_results/evil_genie.pickle.model-tinker___1e1e6607-7cc8-57a8-ae7f-21745560215b_train_0_sampler_weights_000144",
    'eval_results/evil_genie.pickle.model-tinker___51cd023a-e8dd-5b9d-98ca-90dd26b14ca5_train_0_sampler_weights_000216',
    "eval_results/evil_genie.pickle.model-tinker___51cd023a-e8dd-5b9d-98ca-90dd26b14ca5_train_0_sampler_weights_000288",
    "eval_results/evil_genie.pickle.model-tinker___51cd023a-e8dd-5b9d-98ca-90dd26b14ca5_train_0_sampler_weights_000352",
    "eval_results/evil_genie.pickle.model-tinker___51cd023a-e8dd-5b9d-98ca-90dd26b14ca5_train_0_sampler_weights_000400",
    "eval_results/evil_genie.pickle.model-tinker___d5d4218a-e803-5094-90ba-0044afeea523_train_0_sampler_weights_base",
]:
    with open(filename, "rb") as f:
        x = pickle.load(f)
    y = {}
    for (dataset, model), results in x.items():
        model = model.split("/")[-1]
        y[model, dataset] = results
    with open(filename, "wb") as f:
        pickle.dump(y, f)