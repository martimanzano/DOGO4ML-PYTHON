from trust.metrics.metric import Metric
from uq360.metrics.classification_metrics import expected_calibration_error, multiclass_brier_score

class InvertedBrierExpectedCalibration(Metric):
    """Common code used to compute the inverted brier score and the inverted expected calibrated error. For details on each metric, check the corresponding metric's class.

    Args:
        Metric (Class): Metric interface
    """
    
    def assess(trustable_entity):
        print("TRUST - Computing inverted brier/Expected calibration metrics...")
        prediction = trustable_entity.trained_model.predict(trustable_entity.data_x)
        prediction_proba = trustable_entity.trained_model.predict_proba(trustable_entity.data_x)
        
        brier_score = multiclass_brier_score(trustable_entity.data_y, prediction_proba) 
        expected_cal_error = expected_calibration_error(trustable_entity.data_y, prediction_proba, prediction, len(set(trustable_entity.data_y)), False)

        return (1-brier_score), (1-expected_cal_error)
