import pickle

for filename in [
    "eval_results/evil_genie.pickle.model-tinker___d5d4218a-e803-5094-90ba-0044afeea523_train_0_sampler_weights_base",
    "eval_results/evil_genie.pickle.model-tinker___1e1e6607-7cc8-57a8-ae7f-21745560215b_train_0_sampler_weights_000072",
    "eval_results/evil_genie.pickle.model-tinker___1e1e6607-7cc8-57a8-ae7f-21745560215b_train_0_sampler_weights_000144",
]:
    with open(filename, "rb") as f:
        results = pickle.load(f)
    results = {(model.split("/")[-1], task): result for (task, model), result in results.items()}
    print(list(results.keys()))
    with open(filename, "wb") as f:
        pickle.dump(results, f)
