# Group 13

from env_root import *
from additional_metics import *
#from wmd_master import SMD_scorer



### LIBRARY VARS ###



### METRICS ###

# cant run bart, resue and sdc at the same time, requires more than 8GB+ gpu memory
metrics = metrics = {
    "sacrebleu": sacrebleu_score_formatted().compute,
    "meteor": meteor_score_formatted().compute,
    "bart": BART_Score_Eval().compute,
    "reuse": REUSE_score().compute,
    #"sdc*": SDC_Star().compute,            # Slow              
    #'smd': SMD_scorer.calculate_score      # Linux only
}