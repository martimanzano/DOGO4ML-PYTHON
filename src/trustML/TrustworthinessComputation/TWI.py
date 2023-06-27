class TWI:
    """
    Trustworthiness indicator class containing its score and associated metrics and assessment method
    """
        
    def __init__(self):
        self.metrics = None
        self.assessmentMethod = None
        self.TWS = "N/D"
        self.trainedModel = None
        self.dataX = None
        self.dataY = None
    
    def assess(self, trainedModel, dataX, dataY):     
        """Performs the metrics' assessments, followed by the trust assessment
        based on such metrics and the specified assessment method and its parameters.

        Args:
            trainedModel (sklearn classifier): classifier to evaluate
            dataX (pandas dataset): predictor data of the dataset to evaluate
            dataY (pandas dataset): target values of the dataset to evaluate
        """
        self.trainedModel = trainedModel
        self.dataX = dataX
        self.dataY = dataY

        for metric in self.metrics:
            metric.assess(self.trainedModel, self.dataX, self.dataY)

        self.TWS = self.assessmentMethod.assess()

    
    def getTWS(self) -> str:
        """Returns the computed TWS assessment as a string formatted as a JSON document
        
        Returns:
            str: Trustworthiness score formated as a JSON string
        """
        return self.TWS
    
    def getMetricsAssessmentDict(self) -> dict:
        """Returns a dictionary of shape Metric name (str) -> Metric assessment (float)

        Returns:
            dict: Metrics' assessments
        """
        metricNames = [metric.__class__.__name__ for metric in self.metrics]
        metricAssessments = [metric.getValue() for metric in self.metrics]

        return dict(zip(metricNames, metricAssessments))