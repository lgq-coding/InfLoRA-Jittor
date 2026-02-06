
from methods.inflora import InfLoRA


def get_model(model_name, args):
    name = model_name.lower()
    options = {
            #     'sprompts_coda': SPrompts_coda,
            #    'sprompts_l2p': SPrompts_l2p,
            #    'sprompts_dual': SPrompts_dual,
            #    'inflorab5': InfLoRAb5,
               'inflora': InfLoRA,
            #    'inflora_domain': InfLoRA_domain,
            #    'inflorab5_domain': InfLoRAb5_domain,
            #    'inflora_ca': InfLoRA_CA,
            #    'inflora_ca1': InfLoRA_CA1,
               }
    return options[name](args)

