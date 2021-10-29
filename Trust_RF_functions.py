from logging import exception
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklego.metrics import p_percent_score
from aix360.metrics import faithfulness_metric, monotonicity_metric
from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification import SklearnClassifier
import numpy as np
import requests

import pandas as pd

def compute_performance(rf_model, data_x, data_y):
    # USE THE INPUT MODEL TO PREDICT THE TEST SET AND REPORT PERFORMANCE METRICS
    prediction = rf_model.predict(data_x)
    report_dict = classification_report(data_y, prediction, output_dict=True)
    return report_dict 

def compute_fairness(rf_model, data_x, protected_attrs):
    # MODEL P-PERCENTAGE
    p_percentage_vector = np.zeros(data_x.values.shape[0])
    for i in range(len(protected_attrs)):
        p_percentage_vector[i] = p_percent_score(sensitive_column=protected_attrs[i])(rf_model, data_x)
    return np.mean(p_percentage_vector)

def compute_robustness(rf_model, data_x, data_y):
    # ROBUSTNESS VERIFICATION TREE MODELS CLIQUE METHOD (ART)
    rf_skmodel = SklearnClassifier(model=rf_model)
    rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=rf_skmodel)
    average_bound, verified_error = rt.verify(x=data_x.values, y=pd.get_dummies(data_y).values, eps_init=0.3)#, eps_init=0.3, nb_search_steps=10, max_clique=2, 
                                         # max_level=2)
    return average_bound, (1-verified_error) # THE LARGER THE AVRG. BOUND THE BETTER, THE LOWER THE VERIFIED ERROR THE BETTER (SO WE INVERT THE ERROR)

def compute_explainability(rf_model, lime_explainer, data_x):
    ### EXPLAINABILITY ###
    ncases = data_x.values.shape[0]
    monotonicity_vector = np.zeros(ncases) 
    faithfulness_vector = np.zeros(ncases)
    for i in range(ncases):
        predicted_class = rf_model.predict(data_x.values[i].reshape(1,-1))[0]
        explanation = lime_explainer.explain_instance(data_x.values[i], rf_model.predict_proba, num_features=5, top_labels=1)
        local_explanation = explanation.local_exp[predicted_class]

        x = data_x.values[i]
        coefs = np.zeros(x.shape[0])
    
        for v in local_explanation:
            coefs[v[0]] = v[1]
        base = np.zeros(x.shape[0])

        monotonicity_vector[i] = monotonicity_metric(rf_model, data_x.values[i], coefs, base)
        faithfulness_vector[i] = faithfulness_metric(rf_model, data_x.values[i], coefs, base)
    scaler = MinMaxScaler()
    faithfulness_vector_scaled = scaler.fit_transform(faithfulness_vector.reshape(-1,1)) # COMPUTE FROM -1 TO 1, WE SCALED IT TO 0-1 WITH MINMAX    
    return np.mean(monotonicity_vector), np.mean(faithfulness_vector_scaled)

def compute_trust_wrapper(yaml_config, rf_model, data_x, data_y, lime_explainer):
    assessment_method = yaml_config['general']['method']
    if assessment_method == 'bn':
        return compute_trust_BN(yaml_config, rf_model, data_x, data_y, lime_explainer)
    else:
        return compute_trust_weigthed_average(yaml_config, rf_model, data_x, data_y, lime_explainer)

def compute_trust_weigthed_average(yaml_config, rf_model, data_x, data_y, lime_explainer):    
    # PERFORMANCE #
    weight_performance = yaml_config['performance']['weight']
    weight_fairness = yaml_config['fairness']['weight']
    weight_robustness = yaml_config['robustness']['weight']
    weight_explainability = yaml_config['explainability']['weight']

    if weight_performance + weight_fairness + weight_robustness + weight_explainability != 1:
        raise ValueError('Factor weights do not add 1. Revise the configuration file')

    performance_score = perf_accuracy = perf_precision = perf_recall = perf_f1 = 0
    fairness_score = fair_p_percentage = 0
    robustness_score = rob_average_bound = rob_verified_inv_error = 0
    explainability_score = expl_average_monotonicity = expl_average_faithfulness = 0

    if weight_performance > 0:
        perf_dict = compute_performance(rf_model, data_x, data_y)
        perf_accuracy = perf_dict['accuracy'] * yaml_config['performance']['perf_metrics']['accuracy_weight']
        perf_precision = perf_dict['weighted avg']['precision'] * yaml_config['performance']['perf_metrics']['precision_weight']
        perf_recall = perf_dict['weighted avg']['recall'] * yaml_config['performance']['perf_metrics']['recall_weight']
        perf_f1 = perf_dict['weighted avg']['f1-score'] * yaml_config['performance']['perf_metrics']['f1_weight']
        performance_score = (perf_accuracy+perf_precision+perf_recall+perf_f1)*weight_performance
        print("PERFORMANCE SCORE = " + repr(performance_score))
    if weight_fairness > 0:
        protected_attributes = yaml_config['general']['protected_attributes']
        fair_p_percentage = compute_fairness(rf_model, data_x, protected_attributes) * yaml_config['fairness']['fair_metrics']['p-percentage_weight']
        fairness_score = fair_p_percentage*weight_fairness
        print("FAIRNESS SCORE = " + repr(fairness_score))
    if weight_robustness > 0:
        rob_average_bound, rob_verified_inv_error = compute_robustness(rf_model, data_x, data_y)
        rob_average_bound *= yaml_config['robustness']['rob_metrics']['average_bound_weight']
        rob_verified_inv_error *= yaml_config['robustness']['rob_metrics']['verified_error_inv_weight']
        robustness_score = (rob_average_bound+rob_verified_inv_error)*weight_robustness
        print("ROBUSTNESS SCORE = " + repr(robustness_score))
    if weight_explainability > 0:
        expl_average_monotonicity, expl_average_faithfulness = compute_explainability(rf_model, lime_explainer, data_x)
        expl_average_monotonicity *= yaml_config['explainability']['expl_metrics']['average_monotonicity_weight']
        expl_average_faithfulness *= yaml_config['explainability']['expl_metrics']['average_faithfulness_weight'] 
        explainability_score = (expl_average_monotonicity+expl_average_faithfulness)*weight_explainability
        print("EXPLAINABILITY SCORE = " + repr(explainability_score))
    return performance_score+fairness_score+robustness_score+explainability_score

def compute_trust_BN(yaml_config, rf_model, data_x, data_y, lime_explainer):
    protected_attributes = yaml_config['general']['protected_attributes']
    bn_path = yaml_config['general']['bn_parameters']['bn_path']
    api_url = yaml_config['general']['bn_parameters']['api_url']
    id_si = yaml_config['general']['bn_parameters']['id_si']

    input_names = [k for d in yaml_config['general']['bn_parameters']['intervals_input_nodes'] for k in d.keys()]
    intervals_input_nodes = [k for d in yaml_config['general']['bn_parameters']['intervals_input_nodes'] for k in d.values()]

    perf_accuracy = perf_precision = perf_recall = perf_f1 = fair_p_percentage = rob_average_bound = rob_verified_inv_error = expl_average_monotonicity = expl_average_faithfulness = 0
    perf_dict = compute_performance(rf_model, data_x, data_y)
    perf_accuracy = perf_dict['accuracy']
    perf_precision = perf_dict['weighted avg']['precision']
    perf_recall = perf_dict['weighted avg']['recall']
    perf_f1 = perf_dict['weighted avg']['f1-score']
    fair_p_percentage = compute_fairness(rf_model, data_x, protected_attributes)
    rob_average_bound, rob_verified_inv_error = compute_robustness(rf_model, data_x, data_y)
    expl_average_monotonicity, expl_average_faithfulness = compute_explainability(rf_model, lime_explainer, data_x)

    input_values = [perf_accuracy, perf_precision, perf_recall, perf_f1, fair_p_percentage, rob_average_bound, rob_verified_inv_error, expl_average_monotonicity, expl_average_faithfulness]
   
    with open(bn_path, 'rb') as f:
        api_response = requests.post(url=api_url,
                        json={'id_si': id_si, 'input_names': input_names,'input_values': input_values, 'intervals_input_nodes': intervals_input_nodes, 'bn_path': bn_path})
    return api_response.content
    