from nbconvert.preprocessors import *
import re

class ImageLinkPreprocessor(Preprocessor):
    """Strips 'files/' from image links"""

    def preprocess_cell(self, cell, resources, index):
        """Strips 'files/' from image links"""
        MATCHING = 'files\/images'
        if 'source' in cell and cell.cell_type == "markdown":
            cell.source = cell.source.replace('files/images','images')
        
        return cell, resources
