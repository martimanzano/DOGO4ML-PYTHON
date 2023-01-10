from trust.metrics.common.inverted_brier_expected_calibration import InvertedBrierExpectedCalibration


class InvertedBrierSKL(InvertedBrierExpectedCalibration):
    """Inverted brier score metric of a sklearn classifier using UQ360.    

    As it shares code and computation with the inverted expected calibration error metric, it inherits the parent class InvertedBrierExpectedCalibration and leverages the computation of both metrics to this class. It caches the inverted expected calibration error metric metric into the precomputed metrics dict to reduce computation time.   
    
    Args:
        InvertedBrierExpectedCalibration (Class): InvertedBrierExpectedCalibration interface (common code to various metrics)
    """
    
    def assess(trustable_entity):
        if __class__.__name__ in trustable_entity.precomputed_metrics:
            return trustable_entity.precomputed_metrics[__class__.__name__]
        else:
            inverted_brier_score, inverted_expected_cal_error = super(InvertedBrierSKL, InvertedBrierSKL).assess(trustable_entity)

            from trust.metrics.inverted_expected_calibration_uncert_skl import InvertedExpectedCalibrationSKL
            trustable_entity.precomputed_metrics[InvertedExpectedCalibrationSKL.__name__] = inverted_expected_cal_error
            return inverted_brier_score


    # def assess(trustable_entity):
    #     print("TRUST - Computing inverted brier uncertainty metric...")
    #     prediction = trustable_entity.trained_model.predict(trustable_entity.data_x)
    #     prediction_proba = trustable_entity.trained_model.predict_proba(trustable_entity.data_x)
        
    #     brier_score = multiclass_brier_score(trustable_entity.data_y, prediction_proba)        

    #     return (1-brier_score)