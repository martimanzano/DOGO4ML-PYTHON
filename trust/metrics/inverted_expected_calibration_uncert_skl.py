from trust.metrics.common.inverted_brier_expected_calibration import InvertedBrierExpectedCalibration


class InvertedExpectedCalibrationSKL(InvertedBrierExpectedCalibration):
    """Inverted brier score metric of a sklearn classifier using UQ360.    

    As it shares code and computation with the inverted brier score metric, it inherits the parent class InvertedBrierExpectedCalibration and leverages the computation of both metrics to this class. It caches the inverted brier score metric metric into the precomputed metrics dict to reduce computation time.   
    
    Args:
        InvertedBrierExpectedCalibration (Class): InvertedBrierExpectedCalibration interface (common code to various metrics)
    """
    
    def assess(trustable_entity):
        if __class__.__name__ in trustable_entity.precomputed_metrics:
            return trustable_entity.precomputed_metrics[__class__.__name__]
        else:
            inverted_brier_score, inverted_expected_cal_error = super(InvertedExpectedCalibrationSKL, InvertedExpectedCalibrationSKL).assess(trustable_entity)

            from trust.metrics.inverted_brier_uncert_skl import InvertedBrierSKL
            trustable_entity.precomputed_metrics[InvertedBrierSKL.__name__] = inverted_brier_score
            return inverted_expected_cal_error

    # def assess(trustable_entity):
    #     print("TRUST - Computing expected calibration uncertainty metric...")
    #     prediction = trustable_entity.trained_model.predict(trustable_entity.data_x)
    #     prediction_proba = trustable_entity.trained_model.predict_proba(trustable_entity.data_x)
        
    #     expected_cal_error = expected_calibration_error(trustable_entity.data_y, prediction_proba, prediction, len(set(trustable_entity.data_y)), False)

    #     return (1-expected_cal_error)
