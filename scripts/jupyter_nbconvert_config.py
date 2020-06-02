import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Configuration file for jupyter-nbconvert.
c.NbConvertApp.export_format = 'markdown'
c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
c.TagRemovePreprocessor.enabled = True
c.FileLinkPreprocessor.enabled = True
c.ImageLinkPreprocessor.enabled = True
c.Exporter.default_preprocessors = ['nbconvert.preprocessors.TagRemovePreprocessor', 'file_link_preprocessor.FileLinkPreprocessor', 'image_link_preprocessor.ImageLinkPreprocessor']
c.NbConvertApp.postprocessor_class = 'lab_postprocessor.LabPostProcessor'
