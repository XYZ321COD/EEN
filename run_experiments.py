from subprocess import call
# call(["python", "run.py", "-dataset-name", "cifar10", "--arch","resnet50","--b","128", "--epoch", "5", "--save_feature_map", "--load_model", "--save_point", "./runs/Sep30_23-18-33_DESKTOP-300CBSN"])
# call(["python", "run.py", "-dataset-name", "cifar10", "--arch","resnet50","--b","128", "--epoch", "5", "--accm", "--save_feature_map", "--load_model", "--save_point", "./runs/Sep30_23-18-33_DESKTOP-300CBSN", "--h_channels", "32", "--number_of_rsacm", "1"])
# call(["python", "run.py", "-dataset-name", "cifar10", "--arch","resnet50","--b","128", "--epoch", "5", "--accm", "--save_feature_map", "--load_model", "--save_point", "./runs/Sep30_23-18-33_DESKTOP-300CBSN", "--h_channels", "32", "--number_of_rsacm", "2"])
# call(["python", "run.py", "-dataset-name", "cifar10", "--arch","resnet50","--b","128", "--epoch", "5", "--accm", "--save_feature_map", "--load_model", "--save_point", "./runs/Sep30_23-18-33_DESKTOP-300CBSN", "--h_channels", "32", "--number_of_rsacm", "3"])
# call(["python", "run.py", "-dataset-name", "cifar10", "--b","128", "--epoch", "50", "--number_of_rsacm", "1", "--distill_type", "per_accm"])
# call(["python", "run.py", "-dataset-name", "cifar10", "--b","128", "--epoch", "50", "--number_of_rsacm", "1", "--distill_type", "per_rsacm"])
# call(["python", "run.py", "-dataset-name", "cifar10", "--b","128", "--epoch", "50", "--number_of_rsacm", "8", "--distill_type", "per_accm"])
# call(["python", "run.py", "-dataset-name", "cifar10", "--b", "256", "--epoch", "35", "--number_of_rsacm", "8", "--distill_type", "per_rsacm", '--arch', 'resnet50'])
call(["python", "run.py", "-dataset-name", "cifar10", "--b", "128", "--epoch", "35", "--number_of_rsacm", "8", "--distill_type", "per_rsacm", '--arch', 'resnet50', '--load_student'])

# call(["python", "run.py", "-dataset-name", "cifar10", "--b", "128", "--epoch", "100", "--arch", "resnet50"])

