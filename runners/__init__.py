"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.runners.off_policy_ma_runner import OffPolicyMARunner
from harl.runners.off_policy_macpro_runner import OffPolicyMacproRunner

from harl.runners.on_policy_ha_runner_con import OnPolicyHARunnerCon

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "hatd3": OffPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "had3qn": OffPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "matd3": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,

    "macpro": OffPolicyMacproRunner,

    # "happo_ewc": OnPolicyHARunnerCon,
    # "happo_si": OnPolicyHARunnerCon,
    # "happo_mas": OnPolicyHARunnerCon,
    "happo_ewc": OnPolicyHARunner,
    "happo_si": OnPolicyHARunner,
    "happo_mas": OnPolicyHARunner,

    "happo_ewcb": OnPolicyHARunner,
    "happo_sib": OnPolicyHARunner,
    "happo_masb": OnPolicyHARunner,

    "happo_ewcf": OnPolicyHARunner,

    "happo_mh": OnPolicyHARunner,
    "happo_mh2": OnPolicyHARunner,
    "happo_mh3": OnPolicyHARunner,
    "happo_mh5": OnPolicyHARunner,
    "happo_mh6": OnPolicyHARunner,
    "happo_mh7": OnPolicyHARunner,
    "happo_mh8": OnPolicyHARunner,
    "happo_mh9": OnPolicyHARunner,
    "happo_mh10": OnPolicyHARunner,
    "happo_mh11": OnPolicyHARunner,
    "happo_mh12": OnPolicyHARunner,
    "happo_mh13": OnPolicyHARunner,
    "happo_mh14": OnPolicyHARunner,
    "happo_mh15": OnPolicyHARunner,
    "happo_mh16": OnPolicyHARunner,
    "happo_mh17": OnPolicyHARunner,

    "happo_mh18": OnPolicyHARunner,
    "happo_mh19": OnPolicyHARunner,
    "happo_mh20": OnPolicyHARunner,
    "happo_mh21": OnPolicyHARunner,
    "happo_mh22": OnPolicyHARunner,
    "happo_mh23": OnPolicyHARunner,


    "happo_mh24": OnPolicyHARunner,
    "happo_mh25": OnPolicyHARunner,
    "happo_mh26": OnPolicyHARunner,
    "happo_mh27": OnPolicyHARunner,
    "happo_mh28": OnPolicyHARunner,
    "happo_mh29": OnPolicyHARunner,


    "happo_ewc_mh": OnPolicyHARunnerCon,
}
