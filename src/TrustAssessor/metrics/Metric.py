class Metric:
    """Metric interface
    """

    def __init__(self):
        self.assessment = None
    
    def assess(self, trainedModel, classificationModel, dataX, dataY):
        """Assessment of the metric using the trustable entity passed as parameter. It returns a float corresponding to the metric's assessment"""
        
        pass
        #self.assessment = None
    
    def getAssessment(self) -> float:
        return self.assessment