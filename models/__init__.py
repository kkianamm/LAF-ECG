from .medtsllm import MedTsLLM
from .gpt4ts import GPT4TS

from .dlinear import DLinear
from .FEDformer import FEDformer
from .PatchTST import PatchTST
from .TimesNet import TimesNet


model_lookup = {
	"timellm": MedTsLLM,
    "medtsllm": MedTsLLM,
	"gpt4ts": GPT4TS,
    "dlinear": DLinear,
    "fedformer": FEDformer,
    "patchtst": PatchTST,
    "timesnet": TimesNet,
}

# MOMENT-MedTsLLM model registration
from .moment_medtsllm import MomentMedTsLLM
model_lookup["moment_medtsllm"] = MomentMedTsLLM
