from trust.metrics.common.ppercentage_equal_opportunity import PPercentageEqualOppportunity


class PPercentageSKL(PPercentageEqualOppportunity):
    """p_percent metric of a sklearn-based classifier. The p_percent score calculates the ratio between the probability of a positive outcome given the sensitive attribute (column) being true and the same probability given the sensitive attribute being false.

    This is especially useful to use in situations where "fairness" is a theme.

    (Extracted from sklego documentation)

    The metric is computed for the classification model of the TrustableEntity, along with its data_x and data_y. The metric is computed using the protected attributes specified by the user in the configuration YAML/JSON

    As it shares code and computation with the equal opportunity metric, it inherits the parent class PPercentageEqualOppportunity and leverages the computation of both metrics to this class. It caches the equal opportunity metric into the precomputed metrics dict to reduce computation time.

    Args:
        PPercentageEqualOppportunity (Class): PPercentageEqualOppportunity interface (common code to various metrics)
    """
    
    def assess(trustable_entity):
        if __class__.__name__ in trustable_entity.precomputed_metrics:
            return trustable_entity.precomputed_metrics[__class__.__name__]
        else:
            ppercentage, equal_opportunity = super(PPercentageSKL, PPercentageSKL).assess(trustable_entity)

            from trust.metrics.equal_opportunity_skl import EqualOpportunitySKL
            trustable_entity.precomputed_metrics[EqualOpportunitySKL.__name__] = equal_opportunity
            return ppercentage
    
    # def assess2(trustable_entity):
    #     print("TRUST - Computing p-percentage vector...")
    #     if (trustable_entity.protected_attributes is None):
    #         return 1
    #     else:
    #         p_percentage_vector = np.zeros(len(trustable_entity.protected_attributes))#np.zeros(data_x.values.shape[0])
    #         for i in range(len(trustable_entity.protected_attributes)):
    #             p_percentage_vector[i] = p_percent_score(sensitive_column=trustable_entity.protected_attributes[i])(trustable_entity.trained_model, trustable_entity.data_x)
                                  
    #         return np.mean(p_percentage_vector)
