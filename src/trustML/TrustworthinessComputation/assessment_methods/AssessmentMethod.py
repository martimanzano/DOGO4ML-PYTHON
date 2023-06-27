class AssessmentMethod:
    """Assessment method class. Implemented assessment methods should inherit
    this class and implement their custom constructor and assess methods.
    """
    def __init__(self):
        """Initializes the relevant instance's parameters"""
        pass

    def assess(self) -> str:
        """Performs the trustworthiness assessment using the assessment method (child class), and returns it
        """
        pass