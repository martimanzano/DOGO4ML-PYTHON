from trust.Trustable_entity import TrustableEntity
from trust.assessment_methods.assessment_interface import AssessmentInterface
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
from math import isclose

class Trust_Weighted_Average(AssessmentInterface):
    """Class corresponding to the Trust assessment method as a weighted average. It uses the weights and hierarchy specified
    in the YAML/JSON (previously loaded into the TrustableEntity). Therefore, it takes an instance of a TrustableEntity object,
    containing the trustworthiness metrics computed and unweighted."""
    
    def __init__(self, trustable_entity : TrustableEntity):
        """A Trust_WA initializer, corresponding to a specific TrustableEntity

        Args:
            trustable_entity (TrustableEntity): Object of type TrustableEntity for which assess the Trust as a weighted average.
            It must have its trustworthiness metrics already computed as this class is only in charge of the weighted aggregations.
        """
        self.trustable_entity_instance = trustable_entity
        self.assessment = None
        self.trust_hiearchy_node = None
    
    def assess(self):
        """Assesses the TrustEntity as a weighted average and stores the assessment. To ensure the assessment's explainability and traceability,
        this method produces a tree containing the hierarchical assessment, with the raw scores, the weights read from the TrustEntity's configuration,
        and the weighted scores. It leverages the tree computation to the "evaluate_tree" function.
        """
        trust_hiearchy_node = Node(name="Trust")
        self.assessment = self.evaluate_tree(self.trustable_entity_instance.config['w.avg_parameters'], trust_hiearchy_node)
        trust_hiearchy_node.weighted_score = trust_hiearchy_node.raw_score = round(self.assessment, 2)
        trust_hiearchy_node.weight = 1
        self.trust_hiearchy_node = trust_hiearchy_node

    def get_assessment(self):
        """Getter for the WA assessment variable"""
        return self.assessment

    def evaluate_tree(self, dict_node, hiearchy_parent_node):
        """Recursive function that validates and assesses a certain level of the trust hierarchy as a weighted average.
        When such level is not a leaf node, the assessment is performed recursively. When it is a leaf node, the assessment
        is performed by taking the raw metric's value (i.e., score) and the specified weight.

        Args:
            dict_node (dict): Current tree's level to evaluate
            hiearchy_parent_node (Node): Parent node, used to link the current evaluated node with its parent

        Raises:
            Exception: When the hierarchy's weights are not consistent, i.e., any level of the tree does not add up to 1.

        Returns:
            float: weighted score for a hierarchy level (recursively).
        """
        tree_score = node_accumulated_weight = 0
        for node in dict_node:
            node_score = node_weight = 0
            hiearchy_child_node = Node(name=node, parent=hiearchy_parent_node)
            if type(dict_node[node]) is dict:
                hiearchy_child_node.name = self.get_factor_name(node)
                node_weight = self.get_factor_weight(node)
                hiearchy_child_node.weight = node_weight
                node_raw_score = self.evaluate_tree(dict_node[node], hiearchy_child_node)
            else:
                node_weight = dict_node[node]
                hiearchy_child_node.weight = node_weight
                node_raw_score = self.trustable_entity_instance.metrics_assessments[node]
            node_score = node_weight * node_raw_score
            tree_score += node_score
            hiearchy_child_node.weighted_score = round(node_score, ndigits=2)
            hiearchy_child_node.raw_score = round(node_raw_score, ndigits=2)
            node_accumulated_weight += node_weight
        if isclose(node_accumulated_weight, 1):
            return tree_score
        else:
            raise Exception("Validation error in configuration file: weights do not add up to 1 (" + str(node_accumulated_weight) + ") in: \n" + str(dict_node))
    
    def get_factor_weight(self, node):
        return float(node.split("-")[1])
    
    def get_factor_name(self, node):
        return node.split("-")[0]

    def print_weighted_scores_tree(self):
        """Formatted print of the assessment's tree.
        """
        for pre, fill, node in RenderTree(self.trust_hiearchy_node):            
            print("%s%s: %s (raw:%s, weight:%s)" % (pre, node.name, node.weighted_score, node.raw_score, node.weight))
    
    def get_weighted_scores_tree_as_JSON(self):
        """Function that transforms and returns the assessment's tree to a JSON. It leverages on the Anytree package.

        Returns:
            String: Assessment's JSON formatted as a String
        """
        exporter = JsonExporter(indent=2)
        return exporter.export(self.trust_hiearchy_node)