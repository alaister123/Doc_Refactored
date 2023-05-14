import os
import warnings
import pandas
import sys
import datetime

path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(path, "results")

    

def init():
    warnings.filterwarnings(
        action="ignore",
        message="You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset",
        category=UserWarning
    )
    warnings.simplefilter(
        action="ignore",
        category=pandas.errors.PerformanceWarning
    )


def create_result_path():
    if not os.path.exists(result_path):
        os.makedirs(result_path)


def update_metrics():

    common_exp_config = {
    "nlg_metrics" : {},
    "corr_metrics" : ["spearmanr", "pearsonr", "kendalltau"],
    "approaches": ["trad", "new"],
    "eval_levels": ["summary", "system"],
    "result_path_root": "./results/",
    "debug": False
    }

    common_exp_config['nlg_metrics'] = env.metrics
    return common_exp_config
    



def eval(common_exp_config, datapath):
    create_result_path()

    
    summeval_config = {
    "dataset_name": "summeval",
    "human_metrics": ["consistency", "relevance", "coherence", "fluency"],
    "docID_column": "id",
    "document_column": "ArticleText",
    "system_summary_column": "SystemSummary",
    "reference_summary_column": "ReferenceSummary_0",  # the id ranges from 0 to 10
    "is_multi": False, # must be False for SummEval
    "data_path": os.path.join(datapath, "dataloader", "summeval_annotations.aligned.paired.scored.jsonl"),    
    "precalc_metrics": [  # keys from original SummEval json file
        'rouge_1_precision', 'rouge_1_recall', 'rouge_1_f_score',
        'rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score',
        'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f_score',
        'rouge_we_1_p', 'rouge_we_1_r', 'rouge_we_1_f',
        'rouge_we_2_p', 'rouge_we_2_r', 'rouge_we_2_f',
        'meteor', 'cider', 's3_pyr', 's3_resp',
        'mover_score', 'sentence_movers_glove_sms', 'bleu',
        'bert_score_precision', 'bert_score_recall', 'bert_score_f1',
        'blanc', 'summaqa_avg_prob', 'summaqa_avg_fscore', 'supert'], 
    "debug": False
}

    realsumm_abs_config = {
        "dataset_name": "realsumm_abs",
        "human_metrics": ["litepyramid_recall"],
        "docID_column": "doc_id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "data_path": os.path.join(datapath, "dataloader", "abs.pkl"),  # you need to get this file. See ReadMe.
        "result_path_root": "./results/",
        "precalc_metrics": ['rouge_1_f_score', 'rouge_2_recall', 'rouge_l_recall', 'rouge_2_precision',
                                    'rouge_2_f_score', 'rouge_1_precision', 'rouge_1_recall', 'rouge_l_precision',
                                    'rouge_l_f_score', 'js-2', 'mover_score', 'bert_recall_score', 'bert_precision_score',
                                    'bert_f_score'],
        "debug": False                    
    }

    newsroom_config = {
        "dataset_name": "newsroom",
        "human_metrics": ["InformativenessRating", "RelevanceRating", "CoherenceRating", "FluencyRating"],
        "docID_column": "ArticleID",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "human_eval_only_path": os.path.join(datapath, "dataloader", "newsroom-human-eval.csv"),  # you need to get this file. See ReadMe.
        "refs_path": os.path.join(datapath, "dataloader", "test.jsonl"),  # you need to get this file. See ReadMe.
        "human_eval_w_refs_path": os.path.join(datapath, "dataloader", "newsroom_human_eval_with_refs.csv"), 
        "precalc_metrics": [],
    }

    tac2010_config = {
        "dataset_name": "tac2010",
        "human_metrics": ["Pyramid", "Linguistic", "Overall"],
        "approaches": ["new"],
        "docID_column": "docsetID",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "data_path": os.path.join(datapath, "dataloader", "TAC2010"),  # This is a folder. See ReadMe.
        "precalc_metrics": [],
        "is_multi": True, # very important for TAC2010, multi-document summarization
        "debug": False
    }
    
    
    summeval_config.update(common_exp_config)
    realsumm_abs_config.update(common_exp_config)
    newsroom_config.update(common_exp_config)
    tac2010_config.update(common_exp_config)


    summeval.main(summeval_config)
    realsumm.main(realsumm_abs_config)

    realsumm_ext_config = realsumm_abs_config
    realsumm_ext_config["dataset_name"] = "realsumm_ext"
    realsumm_ext_config["data_path"] = os.path.join(datapath, "dataloader", "ext.pkl")
    realsumm_ext_config.update(common_exp_config)

    realsumm.main(realsumm_ext_config)
    newsroom.main(newsroom_config)
    #tac2010.main(tac2010_config)
    



if __name__ == '__main__':
    init()
    env_grp_i = 13
    print("[ENV] group " + str(env_grp_i))
    env_path = os.path.join(path, "env_grp", "g" + str(env_grp_i))
    
    
    sys.path.append(env_path)
    
    
    import env
    
    
    # Change these paths
    sys.path.append("C:\\Users\\Alais\\Desktop\\Doc")
    sys.path.append("C:\\Users\\Alais\\Desktop\\Doc\\evalbase")
    datapath = 'C:\\Users\\Alais\\Desktop\\Doc\\evalbase'
    
    from evalbase import *
    

    
    common_exp_config = update_metrics()



    eval(common_exp_config, datapath)
    # # if os.path.exists(result_path):
    # #     os.rename(result_path, os.path.join(path, "results-g{}-{}".format(env_grp_i, datetime.datetime.now().strftime("%y%m%d-%H%M%S"))))
    # # sys.path.remove(env_path)
    # # if "env" in sys.modules:
    # #     del sys.modules["env"]
    # # del env
