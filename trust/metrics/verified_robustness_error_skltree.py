from trust.metrics.common.average_robustness_verified_error import AverageBoundVerifiedError


class VerifiedErrorSKLTree(AverageBoundVerifiedError):
    """Verified error of a decision-tree-based model on the provided dataset (TrustableEntity -> data_x, data_y) using the ART package.

    As it shares code and computation with the average bound metric, it inherits the parent class AverageBoundVerifiedError and leverages the computation of both metrics to this class. It caches the average bound metric into the precomputed metrics dict to reduce computation time.

    (Extracted from sklearn documentation)

    Args:
        AverageBoundVerifiedError (Class): AverageBoundVerifiedError interface (common code to various metrics)
    """
    
    def assess(trustable_entity):
        if __class__.__name__ in trustable_entity.precomputed_metrics:
            return trustable_entity.precomputed_metrics[__class__.__name__]
        else:
            average_bound, verified_error = super(VerifiedErrorSKLTree, VerifiedErrorSKLTree).assess(trustable_entity)

            from trust.metrics.average_robustness_bound_skltree import AverageBoundSKLTree
            trustable_entity.precomputed_metrics[AverageBoundSKLTree.__name__] = average_bound            
            return verified_error

    # def assess(trustable_entity):
    #     print("TRUST - Computing verified error...")
    #     rf_skmodel = SklearnClassifier(model=trustable_entity.trained_model)
    #     rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=rf_skmodel)        
    #     average_bound, verified_error = rt.verify(x=trustable_entity.data_x.values, y=pd.get_dummies(trustable_entity.data_y).values, eps_init=0.001,
    #      nb_search_steps=1, max_clique=2, max_level=1)

    #     return verified_error

