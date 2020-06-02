from nbconvert.postprocessors import *

class LabPostProcessor(PostProcessorBase):
    def postprocess(self,fileName):
        f = open(fileName, "r")
        data = f.read()
        f.close()
        output = data.replace("```python\n!","```bash\n$ ")
        f = open(fileName, "w")
        f.write(output)
        f.close()

