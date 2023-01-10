class AssessmentInterface:
    """Assessment method interface
    """
    def __init__(trustable_entity):
        """Initializes and loads parameters from the trustable entity and its configuration dict"""
        pass

    def assess(self):
        """Assesses the trustable entity using the assessment method (child class) and stores the resulting assessment
        """
        pass
    def get_assessment(self):
        """Returns the computed assessment in a user-friendly format
        """