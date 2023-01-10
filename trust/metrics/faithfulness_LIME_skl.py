from trust.metrics.common.monotonicity_faithfulness_LIME import MonotonicityFaithfulnessLIME


class FaithfulnessLIMESKL(MonotonicityFaithfulnessLIME):
    """Average Faithfulness metric of a sklearn classifier and a LIME explainer using the AIX360 package.

    This metric evaluates the correlation between the importance assigned by the interpretability algorithm to attributes and the effect of each of the attributes on the performance of the predictive model. The higher the importance, the higher should be the effect, and vice versa, The metric evaluates this by incrementally removing each of the attributes deemed important by the interpretability metric, and evaluating the effect on the performance, and then calculating the correlation between the weights (importance) of the attributes and corresponding model performance. [#]_

    (Extracted from AIX360 documentation)

    As it shares code and computation with the monotonicity metric, it inherits the parent class MonotonicityFaithfulnessLIME and leverages the computation of both metrics to this class. It caches the monotonicity metric into the precomputed metrics dict to reduce computation time.   

    It requires the TrustableEntity to have a LIME tabular explainer (Optional parameter explainer in the TrustableEntity initializer).

    Args:
        MonotonicityFaithfulnessLIME (Class): MonotonicityFaithfulnessLIME interface (common code to various metrics)
    """
    
    def assess(trustable_entity):
        if __class__.__name__ in trustable_entity.precomputed_metrics:
            return trustable_entity.precomputed_metrics[__class__.__name__]
        else:
            average_monotonicity, average_faithfulness = super(FaithfulnessLIMESKL, FaithfulnessLIMESKL).assess(trustable_entity)

            from trust.metrics.monotonicity_LIME_skl import MonotonicityLIMESKL
            trustable_entity.precomputed_metrics[MonotonicityLIMESKL.__name__] = average_monotonicity
            return average_faithfulness

    # def assess(trustable_entity):
    #     print("TRUST - Computing faithfulness metric with LIME...")
    #     ncases = trustable_entity.data_x.values.shape[0]     
    #     faithfulness_vector = np.zeros(ncases) 
    #     for i in tqdm.tqdm(range(ncases), desc="Computing explainability"):
    #         #predicted_class = rf_model.predict(data_x.values[i].reshape(1,-1))[0]
    #         explanation = trustable_entity.explainer.explain_instance(
    #             trustable_entity.data_x.values[i], trustable_entity.trained_model.predict_proba, num_features=5, top_labels=1, num_samples=100)
    #         local_explanation = explanation.local_exp[next(iter(explanation.local_exp))]#explanation.local_exp[predicted_class]

    #         x = trustable_entity.data_x.values[i]
    #         coefs = np.zeros(x.shape[0])
        
    #         for v in local_explanation:
    #             coefs[v[0]] = v[1]
    #         base = np.zeros(x.shape[0])

    #         faithfulness_vector[i] = faithfulness_metric(trustable_entity.trained_model, trustable_entity.data_x.values[i], coefs, base)
    #     scaler = MinMaxScaler()
    #     faithfulness_vector_scaled = scaler.fit_transform(faithfulness_vector.reshape(-1,1)) # COMPUTE FROM -1 TO 1, WE SCALED IT TO 0-1 WITH MINMAX

    #     return np.mean(faithfulness_vector_scaled)
    
