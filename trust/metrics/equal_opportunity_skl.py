from trust.metrics.common.ppercentage_equal_opportunity import PPercentageEqualOppportunity


class EqualOpportunitySKL(PPercentageEqualOppportunity):
    """Equal opportunity metric of a sklearn-based classifier. The equality opportunity score calculates the ratio between the probability of a **true positive** outcome given the sensitive attribute (column) being true and the same probability given the sensitive attribute being false.

    This is especially useful to use in situations where "fairness" is a theme.

    (Extracted from sklego documentation)

    The metric is computed for the classification model of the TrustableEntity, along with its data_x and data_y. The metric is computed using the protected attributes specified by the user in the configuration YAML/JSON

    As it shares code and computation with the p-percent score metric, it inherits the parent class PPercentageEqualOppportunity and leverages the computation of both metrics to this class. It caches the p-percent metric into the precomputed metrics dict to reduce computation time.

    Args:
        PPercentageEqualOppportunity (Class): PPercentageEqualOppportunity interface (common code to various metrics)
    """
    
    def assess(trustable_entity):
        if __class__.__name__ in trustable_entity.precomputed_metrics:
            return trustable_entity.precomputed_metrics[__class__.__name__]
        else:
            ppercentage, equal_opportunity = super(EqualOpportunitySKL, EqualOpportunitySKL).assess(trustable_entity)

            from trust.metrics.p_percentage_skl import PPercentageSKL
            trustable_entity.precomputed_metrics[PPercentageSKL.__name__] = ppercentage
            return equal_opportunity

    # def assess(trustable_entity):
    #     print("TRUST - Computing equal opportunity vector...")
    #     if (trustable_entity.protected_attributes is None):
    #         return 1
    #     else:
    #         eq_opp_vector = np.zeros(len(trustable_entity.protected_attributes))
    #         for i in range(len(trustable_entity.protected_attributes)):
    #             eq_opp_vector[i] = equal_opportunity_score(trustable_entity.protected_attributes[i])(trustable_entity.trained_model, trustable_entity.data_x, trustable_entity.data_y)
                                  
    #         return np.mean(eq_opp_vector)
