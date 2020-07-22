from nbconvert.preprocessors import *
import re

class FileLinkPreprocessor(Preprocessor):
    """Fixes file links in markdown"""

    def preprocess_cell(self, cell, resources, index):
        """
        Changes **filename.ext ( File -> Open -> filename.ext ) 
        to a proper link.
        """
        MATCHING = '\*\*.*\*\* \(.*File.*\-\>.*Open.*\-\>.*\)'
        if 'source' in cell and cell.cell_type == "markdown":
            for found in re.findall(MATCHING,cell.source):
                linkname = re.split('\*\*',found)[1]
                filename = re.split('Open.*\-\>',found)[1]
                filename = filename.strip(')')
                filename = filename.strip()
                cell.source = cell.source.replace(found,f'[{linkname}]({filename})')
        
        return cell, resources
