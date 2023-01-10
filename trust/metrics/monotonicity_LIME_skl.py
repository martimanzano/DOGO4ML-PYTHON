from trust.metrics.common.monotonicity_faithfulness_LIME import MonotonicityFaithfulnessLIME


class MonotonicityLIMESKL(MonotonicityFaithfulnessLIME):
    """Average Monotonicity metric of a sklearn classifier and a LIME explainer using the AIX360 package.

    This metric measures the effect of individual features on model performance by evaluating the effect on model performance of incrementally adding each attribute in order of increasing importance. As each feature is added, the performance of the model should correspondingly increase, thereby resulting in monotonically increasing model performance. [#]_

    (Extracted from AIX360 documentation)

    As it shares code and computation with the faithfulness metric, it inherits the parent class MonotonicityFaithfulnessLIME and leverages the computation of both metrics to this class. It caches the faithfulness metric into the precomputed metrics dict to reduce computation time.   

    It requires the TrustableEntity to have a LIME tabular explainer (Optional parameter explainer in the TrustableEntity initializer).

    Args:
        MonotonicityFaithfulnessLIME (Class): MonotonicityFaithfulnessLIME interface (common code to various metrics)
    """
    
    def assess(trustable_entity):
        if __class__.__name__ in trustable_entity.precomputed_metrics:
            return trustable_entity.precomputed_metrics[__class__.__name__]
        else:
            average_monotonicity, average_faithfulness = super(MonotonicityLIMESKL, MonotonicityLIMESKL).assess(trustable_entity)

            from trust.metrics.faithfulness_LIME_skl import FaithfulnessLIMESKL
            trustable_entity.precomputed_metrics[FaithfulnessLIMESKL.__name__] = average_faithfulness
            return average_monotonicity

    # def assess(trustable_entity):
    #     print("TRUST - Computing monotonicity metric with LIME...")
    #     ncases = trustable_entity.data_x.values.shape[0]     
    #     monotonicity_vector = np.zeros(ncases) 
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

    #         monotonicity_vector[i] = monotonicity_metric(trustable_entity.trained_model, trustable_entity.data_x.values[i], coefs, base)

    #     return np.mean(monotonicity_vector)
    
